import os
import time
from time import ctime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from importlib import import_module
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# from torcheval.metrics.aggregation.auc import AUC

import shutil
import warnings
import builtins
import random
import math
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from dataset import ImageDataset
from model.utils import get_model

from utils import AverageMeter, calculate_accuracy, save_checkpoint, adjust_learning_rate, ProgressMeter_wID

from config import create_arg_parser


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training.".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            print("gpu... ", gpu)
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend = args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size = args.world_size,
            rank = args.rank
        )
    
    t = time.time()
    # model = get_model(args, weights=args.pretrained_weights)
    model = get_model(args)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            # args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size = 1
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            generator = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            generator = DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDAtaParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDAtaParallel is supported.")


    print("=> loading checkpoint '{}'".format(args.test_pth+'/checkpoint_best.pth.tar'))
    if args.test_pth[:4] != '/mnt':
        args.test_pth = '/mnt/KW/LungCancer/Multimodality2/results/SavedModels/' + args.test_pth + '/checkpoint_best.pth.tar'
        
    if args.gpu is None:
        checkpoint = torch.load(args.test_pth)
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(args.test_pth, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(
            args.test_pth, checkpoint["epoch"]
        )
    )
    print(f"Time for model load : {time.time()-t:.4f} seconds.")

    cudnn.benchmark = True

    # criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)
    if args.num_classes > 2:
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = torch.nn.BCELoss().cuda(args.gpu)
    
    # if 'textCosSim' in args.loss:
    criterion_CosSim = torch.nn.CosineEmbeddingLoss().cuda(args.gpu)
    
    t = time.time()
    test_dataset = ImageDataset(args, mode=args.mode)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Time for dataset loader load : {time.time()-t:.4f} seconds.")
    
    ######################################################################################################################
    test_loss, test_acc, DF, preds, labels, inf_time = test(dataloader_test, generator, criterion, criterion_CosSim, args)
    ######################################################################################################################
    print(f"Time for inference : {(sum(inf_time[1:])/len(inf_time[1:])):.4f} seconds for each samples.")
    
    save_pth = args.test_pth[:-23].replace('SavedModels', 'Predictions')
    save_pth = save_pth.replace('tr('+args.cancerstageTrain+')', 'tr('+args.cancerstageTrain+')te('+args.cancerstageTest+')')
    # save_pth = save_pth[:-21].replace('tr('+args.cancerstageTrain+')', 'tr('+args.cancerstageTrain+')te('+args.cancerstageTest+')') + save_pth[-21:]
                                    #   )'te(' + args.cancerstageTest + ')' + save_pth[-21:]
    
    if not os.path.exists(save_pth):
        os.makedirs(save_pth, exist_ok=True)
    
    with open(f"{save_pth}config.txt", 'w') as f:
        for key in vars(args).keys():
            f.write(f"{key}: {vars(args)[key]}\n")
    
    test_auc, best_thres = ROC_curve_plot(labels, preds, save_pth[:-1])
    
    
    # test mode 일 때는, valid mode에서 구한 best_thres를 manual하게 적어줘야 함    
    if args.mode == 'test':
        best_thres = args.best_thres
    preds = [(item >= best_thres) for item in preds]
    test_acc = accuracy_score(labels, preds)
    test_recall = recall_score(labels, preds)
    test_precision = precision_score(labels, preds)
    
    epoch = args.test_pth[-12:-8]
    if args.mode == 'valid':
        DF.to_excel(save_pth + 'result_valid%d(%s)_AUC(%.4f)ACC(%.4f)Precision(%.4f)Recall(%.4f)_thres(%.4f).xlsx' %(args.val_fold, epoch, test_auc, test_acc, test_precision, test_recall, best_thres))
    else:
        DF.to_excel(save_pth + 'result_test%d(%s)_AUC(%.4f)ACC(%.4f)Precision(%.4f)Recall(%.4f).xlsx' %(args.val_fold, epoch, test_auc, test_acc, test_precision, test_recall))
    
    print('-------------------------------------------------------------------------------')
    if args.mode == 'valid':
        print("Result : Loss %.4e   AUC %.4f   Accuracy %.4f   Precision %.4f   Recall %.4f | Threshold %.4f" % (test_loss, test_auc, test_acc, test_precision, test_recall, best_thres))
    else:
        print("Result : Loss %.4e   AUC %.4f   Accuracy %.4f   Precision %.4f   Recall %.4f" % (test_loss, test_auc, test_acc, test_precision, test_recall))


def test(dataloader_test, generator, criterion, criterion_CosSim, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    losses_CosSim = AverageMeter("Loss", ":.4e")
    accs = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter_wID(
        len(dataloader_test),
        [batch_time, data_time, losses, accs],
        prefix="Test: ",
    )
    
    # AUC_metric = AUC()
    DF = pd.DataFrame()
    
    preds = []
    labels = []
    inf_time = []
    
    label_for_CosSim = torch.tensor([1], dtype=torch.float32).cuda(args.gpu)
    
    with torch.no_grad():
        generator.eval()

        end = time.time()
        for i, test_data_dict in enumerate(dataloader_test):
            data_time.update(time.time() - end)
            
            if args.gpu is not None:
                if ('CT' in args.modality) & ('pathology' in args.modality):
                    test_input_CT = test_data_dict['input_CT'].cuda(non_blocking=True)
                    test_input_pathology = test_data_dict['input_pathology'].cuda(non_blocking=True)
                elif ('CT' in args.modality):
                    test_input_CT = test_data_dict['input_CT'].cuda(non_blocking=True)
                elif ('pathology' in args.modality):
                    test_input_pathology = test_data_dict['input_pathology'].cuda(non_blocking=True)
                
                test_input_CI = test_data_dict['input_CI'].cuda(non_blocking=True)
                    
                if 'wMask' in args.model_CT:
                    test_mask = test_data_dict['mask'].cuda(non_blocking=True)
                
                if 'pathology' in args.modality:
                    if args.model_pathology == 'ABMIL_v2':
                        test_info_BpRc = test_data_dict['BpRc_class'].float().cuda(non_blocking=True)
                
                test_label = test_data_dict['label'].float().cuda(non_blocking=True)
                test_ID = test_data_dict['ID'][0]
            
            t = time.time()
            if ('CT' in args.modality) & ('pathology' in args.modality):
                if 'wMask' in args.model_CT:
                    test_output, test_CT2CI, test_Pth2CI, _ = generator([test_input_CT, test_input_pathology], test_input_CI, test_mask)
                else:
                    # test_output, test_CT2CI, test_Pth2CI, _ = generator([test_input_CT, test_input_pathology], test_input_CI)
                    test_output, test_CT2CI, test_Pth2CI = generator([test_input_CT, test_input_pathology], test_input_CI)
            elif ('CT' in args.modality):
                if 'wMask' in args.model_CT:
                    test_output, test_CT2CI, _ = generator([test_input_CT], test_input_CI, test_mask)
                else:
                    # test_output, test_CT2CI, _ = generator([test_input_CT], test_input_CI)
                    test_output, _ = generator([test_input_CT], test_input_CI)
            elif ('pathology' in args.modality):
                if args.model_pathology == 'ABMIL_v2':
                    test_output, test_Pth2CI, _ = generator([test_input_pathology, test_info_BpRc], test_input_CI)
                else:
                    test_output, test_Pth2CI, _ = generator([test_input_pathology], test_input_CI)
            elif ('CI' in args.modality):
                test_output, _ = generator([], test_input_CI)
            inf_time.append(time.time() - t)

            test_loss = criterion(test_output, test_label)
            if 'textCosSim' in args.loss:
                test_loss_textCosSim = criterion_CosSim(test_CT2CI.squeeze(1), test_Pth2CI.squeeze(1), label_for_CosSim)
                losses_CosSim.update(test_loss_textCosSim.item(), test_output.size(0))
                
                test_loss += test_loss_textCosSim
            test_acc = calculate_accuracy(test_output, test_label)
            # AUC_metric.update(test_output[:,1], test_label)

            losses.update(test_loss.item(), test_output.size(0))
            accs.update(test_acc.item(), test_output.size(0))
            
            # for b in range(test_output.shape[0]):
            #     # preds.append(torch.argmax(test_output[b,:]).item())
            #     # labels.append(torch.argmax(test_label[b,:]).item())
            #     preds.append(test_output[b,:].tolist())
            #     labels.append(test_label[b,:].tolist())
            preds.append(test_output[:,1].item())
            labels.append(test_label[:,1].item())

            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(test_ID, i)
            
            test_output_ = []
            test_output_.append(test_ID)
            for t in range(test_output.shape[1]):
                test_output_.append(test_output[0,t].item())
            test_output_.append(torch.argmax(test_label[0,:]).item())
            test_output_.append(test_acc.item())
            
            test_output_label = []
            test_output_label.append('ID')
            for t in range(test_output.shape[1]):
                test_output_label.append('Probabiltity_ch%d'%(t))
            test_output_label.append('Label')
            test_output_label.append('Accuracy')
            
            DF = pd.concat([DF, pd.DataFrame(data=[test_output_], index=[i], columns=test_output_label)])
            
            # DF = pd.concat([DF, pd.DataFrame(data=[[test_ID, test_output[:,0].item(), test_output[:,1].item(), test_label[:,1].item(), test_acc.item()]], index=[i], columns=['ID', 'Probability_ch0', 'Probability_ch1', 'Label', 'Accuracy'])])
            # DF = pd.concat([DF, pd.DataFrame(data=[[test_ID, test_output[:,1].item(), test_label[:,1].item(), test_acc.item()]], index=[i], columns=['ID', 'Probability', 'Label', 'Accuracy'])])
        
        # auc = AUC_metric.compute()
    
    # return losses.avg, accs.avg, auc, DF, preds, labels
    return losses.avg, accs.avg, DF, preds, labels, inf_time


from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt

def ROC_curve_plot(label, score, save_path):
    fpr, tpr, thresholds = roc_curve(label, score)
    best_idx = np.argmax(tpr-fpr)
    best_threshold = thresholds[best_idx]
    # auc1 = auc(fpr, tpr)
    auc = roc_auc_score(label, score)

    plt.figure(figsize=(5,5), dpi= 300)
    plt.plot(fpr, tpr, color='#FC5A50', linestyle='-', label=r'AUC=%.4f' %auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('5y OS Prediction')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.legend()
    plt.savefig(save_path + '/ROC.png', dpi=600)
    
    return auc, best_threshold



if __name__ == "__main__":
    args = create_arg_parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


    sys.exit(0)