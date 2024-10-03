import os
import time
from time import ctime

# from sklearnex import patch_sklearn
# patch_sklearn()

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

from torcheval.metrics.aggregation.auc import AUC
from sklearn.metrics import roc_auc_score, precision_score, recall_score

import shutil
import warnings
import builtins
import random
import math
import sys
from shutil import copyfile

from dataset import ImageDataset
from model.utils import get_model
from utils import AverageMeter, calculate_accuracy, save_checkpoint, adjust_learning_rate, ProgressMeter

from config import create_arg_parser


def main_worker(gpu, ngpus_per_node, args, save_dir):
    writer = SummaryWriter(log_dir=save_dir.replace('results/SavedModels', 'runs'))

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
            # args.rank = gpu
        dist.init_process_group(
            backend = args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size = args.world_size,
            rank = args.rank
        )

    # model = get_model(args, weights=args.pretrained_weights)
    model = get_model(args)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            args.batch_size = int(args.batch_size / ngpus_per_node)
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
    
    
    cudnn.benchmark = True

    # criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)
    criterion_CLIP = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    if args.num_classes > 2:
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = torch.nn.BCELoss().cuda(args.gpu)
    
    # if 'textCosSim' in args.loss:
    criterion_CosSim = torch.nn.CosineEmbeddingLoss().cuda(args.gpu)
    
    if args.learnablePrompt:
        args.lr = 0.001
        optimizer = torch.optim.SGD(generator.parameters(),
                                    lr=args.lr,
                                    weight_decay=10**-7
                                    )
    else:
        if args.num_classes > 2:
            args.lr = 0.001
        else:
            args.lr = 0.00001
        optimizer = torch.optim.Adam(generator.parameters(),
                                    lr=args.lr,
                                    betas=(args.b1, args.b2),
                                    weight_decay=10**-7)
    
    if args.resume:
        if args.resume[:4] != '/mnt':
            args.resume = '/mnt/KW/LungCancer/Multimodality2/results/SavedModels/' + args.resume + '/checkpoint_best.pth.tar'
                
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if args.pretrainedExt_freeze:
                if args.pretrainedExt_CT and 'CT' in args.modality:
                    for p in model.extractor_CT.parameters():
                        p.requires_grad = False
                if args.pretrainedExt_pathology and 'pathology' in args.modality:
                    for p in model.extractor_pathology.parameters():
                        p.requires_grad = False
                if args.pretrainedExt_CI and 'CI' in args.modality:
                    for p in model.extractor_CI.parameters():
                        p.requires_grad = False
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.pretrainedExt_CT and 'CT' in args.modality:
            pretrained_CT_extractor = torch.load(args.pretrainedExt_CT_pth)
            pretrained_CT_extractor_dict = pretrained_CT_extractor["state_dict"]
            model_CT_dict = model.extractor_CT.state_dict()
            pretrained_CT_extractor_dict = {k: v for k, v in pretrained_CT_extractor_dict.items() if k in model_CT_dict}
            model_CT_dict.update(pretrained_CT_extractor_dict)
            model.extractor_CT.load_state_dict(model_CT_dict)
            if args.pretrainedExt_freeze:
                for p in model.extractor_CT.parameters():
                    p.requires_grad = False
        
        if args.pretrainedExt_pathology and 'pathology' in args.modality:
            pretrained_pathology_extractor = torch.load(args.pretrainedExt_pathology_pth)
            pretrained_pathology_extractor_dict = pretrained_pathology_extractor["state_dict"]
            model_pathology_dict = model.extractor_pathology.state_dict()
            pretrained_pathology_extractor_dict = {k: v for k, v in pretrained_pathology_extractor_dict.items() if k in model_pathology_dict}
            model_pathology_dict.update(pretrained_pathology_extractor_dict)
            model.extractor_pathology.load_state_dict(model_pathology_dict)
            if args.pretrainedExt_freeze:
                for p in model.extractor_pathology.parameters():
                    p.requires_grad = False
        
        if args.pretrainedExt_CI and 'CI' in args.modality:
            pretrained_CI_extractor = torch.load(args.pretrainedExt_CI_pth)
            pretrained_CI_extractor_dict = pretrained_CI_extractor["state_dict"]
            model_CI_dict = model.extractor_CI.state_dict()
            pretrained_CI_extractor_dict = {k: v for k, v in pretrained_CI_extractor_dict.items() if k in model_CI_dict}
            model_CI_dict.update(pretrained_CI_extractor_dict)
            model.extractor_CI.load_state_dict(model_CI_dict)
            if args.pretrainedExt_freeze:
                for p in model.extractor_CI.parameters():
                    p.requires_grad = False
    

    train_dataset = ImageDataset(args, mode='train')
    valid_dataset = ImageDataset(args, mode='valid')
    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    else:
        train_sampler = None

    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    dataloader_valid = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,                   num_workers=args.num_workers, pin_memory=True)
    
    valid_auc_best = 0
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        
        print(save_dir)
        print("---------------------------------------------------------------------------------------------------------------------------")
        train(dataloader_train, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer)
        print("---------------------------------------------------------------------------------------------------------------------------")
        _, valid_acc, valid_auc = valid(dataloader_valid, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer)
        print("---------------------------------------------------------------------------------------------------------------------------")

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            is_best = args.save_best
            if is_best:
                if valid_auc_best <= valid_auc:
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        is_best=is_best,
                        save_dir = save_dir,
                        filename="checkpoint_{:04d}.pth.tar".format(epoch),
                    )
                    valid_auc_best = valid_auc
            else:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=is_best,
                    save_dir = save_dir,
                    filename="checkpoint_{:04d}.pth.tar".format(epoch),
                )
            torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, save_dir + '/checkpoint_last.pth.tar')

def train(dataloader_train, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    losses_CosSim = AverageMeter("Loss", ":.4e")
    if args.loss_point == 'CT-Pth-Last':
        losses_CT = AverageMeter("Loss_CT", ":.4e")
        losses_Pth = AverageMeter("Loss_Pth", ":.4e")
        losses_Last = AverageMeter("Loss_Last", ":.4e")
    accs = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(dataloader_train),
        [batch_time, data_time, losses, accs],
        prefix="Train Epoch: [{}]".format(epoch),
    )
    
    # AUC_metric = AUC()
    preds = []
    labels = []
    
    label_for_CosSim = torch.tensor([1], dtype=torch.float32).cuda(args.gpu)

    generator.train()

    end = time.time()
    for i, train_data_dict in enumerate(dataloader_train):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if ('CT' in args.modality) & ('pathology' in args.modality):
                train_input_CT = train_data_dict['input_CT'].cuda(non_blocking=True)
                train_input_pathology = train_data_dict['input_pathology'].cuda(non_blocking=True)
            elif ('CT' in args.modality):
                train_input_CT = train_data_dict['input_CT'].cuda(non_blocking=True)
            elif ('pathology' in args.modality):
                train_input_pathology = train_data_dict['input_pathology'].cuda(non_blocking=True)
            
            # train_input_CI = train_data_dict['input_CI'].cuda(non_blocking=True)
            train_input_CI = train_data_dict['input_CI'] # Because it is a list.
                
            if 'wMask' in args.model_CT:
                train_mask = train_data_dict['mask'].cuda(non_blocking=True)
            
            if 'pathology' in args.modality:
                if args.model_pathology == 'ABMIL_v2':
                    train_info_BpRc = train_data_dict['BpRc_class'].float().cuda(non_blocking=True)
            
            train_label = train_data_dict['label'].float().cuda(non_blocking=True)

        if ('CT' in args.modality) & ('pathology' in args.modality):
            if 'wMask' in args.model_CT:
                train_output, train_CT2CI, train_Pth2CI, _ = generator([train_input_CT, train_input_pathology], train_input_CI, train_mask)
            else:
                # [train_output, train_output_CTonly, train_output_Pthonly], [train_CT2CI, train_Pth2CI], [attns, attns_CT, attns_Pth]
                train_outputs, train_CI, _ = generator([train_input_CT, train_input_pathology], train_input_CI)
                # train_output, train_CT2CI, train_Pth2CI, train_CI2CT, train_CI2Pth = generator([train_input_CT, train_input_pathology], train_input_CI)
        elif ('CT' in args.modality):
            if args.alignment_base == 'none':
                train_output, _ = generator([train_input_CT], train_input_CI)
            else:
                if 'wMask' in args.model_CT:
                    train_output, train_CT2CI, _ = generator([train_input_CT], train_input_CI, train_mask)
                else:
                    train_output, train_CT2CI, _ = generator([train_input_CT], train_input_CI)
        elif ('pathology' in args.modality):
            if args.model_pathology == 'ABMIL_v2':
                train_output, train_Pth2CI, _ = generator([train_input_pathology, train_info_BpRc], train_input_CI)
            else:
                train_output, train_Pth2CI, _ = generator([train_input_pathology], train_input_CI)
        elif ('CI' in args.modality):
            train_output, _ = generator([], train_input_CI)
        
        if args.loss_point == 'CT-Pth-Last':
            train_loss_CT = criterion(train_outputs[1], train_label)
            train_loss_Pth = criterion(train_outputs[2], train_label)
            train_loss_Last = criterion(train_outputs[0], train_label)
            train_loss = train_loss_CT + train_loss_Pth + train_loss_Last
        elif args.loss_point == 'Last':
            train_loss = criterion(train_outputs[0], train_label)
        if 'textCosSim' in args.loss:
            train_loss_textCosSim = criterion_CosSim(train_CI[0].squeeze(1), train_CI[1].squeeze(1), label_for_CosSim)
            losses_CosSim.update(train_loss_textCosSim.item(), train_outputs[0].size(0))
            
            train_loss += train_loss_textCosSim
        
        train_acc = calculate_accuracy(train_outputs[0], train_label)

        losses.update(train_loss.item(), train_outputs[0].size(0))
        accs.update(train_acc.item(), train_outputs[0].size(0))
        if args.loss_point == 'CT-Pth-Last':
            losses_CT.update(train_loss_CT.item(), train_outputs[0].size(0))
            losses_Pth.update(train_loss_Pth.item(), train_outputs[0].size(0))
            losses_Last.update(train_loss_Last.item(), train_outputs[0].size(0))
        # AUC_metric.update(train_output[:,0], train_label)
        
        for b in range(train_outputs[0].shape[0]):
            # preds.append(train_output[b,0].item())
            preds.append(torch.argmax(train_outputs[0][b,:]).item())
            labels.append(torch.argmax(train_label[b,:]).item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        if i % 10 == 0:
            progress.display(i)
    
    
    writer.add_scalar('train/loss', losses.avg, epoch)
    if args.loss_point == 'CT-Pth-Last':
        writer.add_scalar('train/loss_CT', losses_CT.avg, epoch)
        writer.add_scalar('train/loss_Pth', losses_Pth.avg, epoch)
        writer.add_scalar('train/loss_Last', losses_Last.avg, epoch)
    writer.add_scalar('train/loss_CosSim', losses_CosSim.avg, epoch)
    writer.add_scalar('train/acc', accs.avg, epoch)
    if args.num_classes > 2:
        train_auc = roc_auc_score(np.eye(5)[labels], np.eye(5)[preds], multi_class='ovo', average='macro')
    else:
        train_auc = roc_auc_score(labels, preds)
    writer.add_scalar('train/auc', train_auc, epoch)
    
    preds = [round(item) for item in preds]
    if args.num_classes > 2:
        train_recall = recall_score(labels, preds, average='macro', zero_division=np.nan)
        train_precision = precision_score(labels, preds, average='macro', zero_division=np.nan)
    else:
        train_recall = recall_score(labels, preds, zero_division=np.nan)
        train_precision = precision_score(labels, preds, zero_division=np.nan)
    writer.add_scalar('train/recall', train_recall, epoch)
    writer.add_scalar('train/precision', train_precision, epoch)


def valid(dataloader_valid, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    losses_CosSim = AverageMeter("Loss", ":.4e")
    if args.loss_point == 'CT-Pth-Last':
        losses_CT = AverageMeter("Loss_CT", ":.4e")
        losses_Pth = AverageMeter("Loss_Pth", ":.4e")
        losses_Last = AverageMeter("Loss_Last", ":.4e")
    accs = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(dataloader_valid),
        [batch_time, data_time, losses, accs],
        prefix="Valid Epoch: [{}]".format(epoch),
    )
    
    # AUC_metric = AUC()
    preds = []
    labels = []
    
    label_for_CosSim = torch.tensor([1], dtype=torch.float32).cuda(args.gpu)
    
    with torch.no_grad():
        generator.eval()

        end = time.time()
        for i, valid_data_dict in enumerate(dataloader_valid):
            data_time.update(time.time() - end)

            if args.gpu is not None:
                if ('CT' in args.modality) & ('pathology' in args.modality):
                    valid_input_CT = valid_data_dict['input_CT'].cuda(non_blocking=True)
                    valid_input_pathology = valid_data_dict['input_pathology'].cuda(non_blocking=True)
                elif ('CT' in args.modality):
                    valid_input_CT = valid_data_dict['input_CT'].cuda(non_blocking=True)
                elif ('pathology' in args.modality):
                    valid_input_pathology = valid_data_dict['input_pathology'].cuda(non_blocking=True)
                    
                # valid_input_CI = valid_data_dict['input_CI'].cuda(non_blocking=True)
                valid_input_CI = valid_data_dict['input_CI']
                    
                if 'wMask' in args.model_CT:
                    valid_mask = valid_data_dict['mask'].cuda(non_blocking=True)
            
                if 'pathology' in args.modality:
                    if args.model_pathology == 'ABMIL_v2':
                        valid_info_BpRc = valid_data_dict['BpRc_class'].float().cuda(non_blocking=True)
                
                valid_label = valid_data_dict['label'].float().cuda(non_blocking=True)

            if ('CT' in args.modality) & ('pathology' in args.modality):
                if 'wMask' in args.model_CT:
                    valid_output, valid_CT2CI, valid_Pth2CI, _ = generator([valid_input_CT, valid_input_pathology], valid_mask)
                else:
                    valid_outputs, valid_CI, _ = generator([valid_input_CT, valid_input_pathology], valid_input_CI)
                    # valid_output, valid_CT2CI, valid_Pth2CI, valid_CI2CT, valid_CI2Pth = generator([valid_input_CT, valid_input_pathology], valid_input_CI)
            elif ('CT' in args.modality):
                if args.alignment_base == 'none':
                    valid_output, _ = generator([valid_input_CT], valid_input_CI)
                else:
                    if 'wMask' in args.model_CT:
                        valid_output, valid_CT2CI, _ = generator([valid_input_CT], valid_input_CI, valid_mask)
                    else:
                        valid_output, valid_CT2CI, _ = generator([valid_input_CT], valid_input_CI)
            elif ('pathology' in args.modality):
                if args.model_pathology == 'ABMIL_v2':
                    valid_output, valid_Pth2CI, _ = generator([valid_input_pathology, valid_info_BpRc], valid_input_CI)
                else:
                    valid_output, valid_Pth2CI, _ = generator([valid_input_pathology], valid_input_CI)
            elif ('CI' in args.modality):
                valid_output, _ = generator([], valid_input_CI)
            
            if args.loss_point == 'CT-Pth-Last':
                valid_loss_CT = criterion(valid_outputs[1], valid_label)
                valid_loss_Pth = criterion(valid_outputs[2], valid_label)
                valid_loss_Last = criterion(valid_outputs[0], valid_label)
                valid_loss = valid_loss_CT + valid_loss_Pth + valid_loss_Last
            elif args.loss_point == 'Last':
                valid_loss = criterion(valid_outputs[0], valid_label)
            if 'textCosSim' in args.loss:
                valid_loss_textCosSim = criterion_CosSim(valid_CI[0].squeeze(1), valid_CI[1].squeeze(1), label_for_CosSim)
                losses_CosSim.update(valid_loss_textCosSim.item(), valid_outputs[0].size(0))
                
                valid_loss += valid_loss_textCosSim
            
            valid_acc = calculate_accuracy(valid_outputs[0], valid_label)

            losses.update(valid_loss.item(), valid_outputs[0].size(0))
            accs.update(valid_acc.item(), valid_outputs[0].size(0))
            if args.loss_point == 'CT-Pth-Last':
                losses_CT.update(valid_loss_CT.item(), valid_outputs[0].size(0))
                losses_Pth.update(valid_loss_Pth.item(), valid_outputs[0].size(0))
                losses_Last.update(valid_loss_Last.item(), valid_outputs[0].size(0))
            # AUC_metric.update(valid_output[:,0], valid_label)
        
            for b in range(valid_outputs[0].shape[0]):
                # preds.append(valid_output[b,0].item())
                preds.append(torch.argmax(valid_outputs[0][b,:]).item())
                labels.append(torch.argmax(valid_label[b,:]).item())

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            if i % 10 == 0:
                progress.display(i)
    
    
    writer.add_scalar('valid/loss', losses.avg, epoch)
    if args.loss_point == 'CT-Pth-Last':
        writer.add_scalar('valid/loss_CT', losses_CT.avg, epoch)
        writer.add_scalar('valid/loss_Pth', losses_Pth.avg, epoch)
        writer.add_scalar('valid/loss_Last', losses_Last.avg, epoch)
    writer.add_scalar('valid/loss_CosSim', losses_CosSim.avg, epoch)
    writer.add_scalar('valid/acc', accs.avg, epoch)
    if args.num_classes > 2:
        valid_auc = roc_auc_score(np.eye(5)[labels], np.eye(5)[preds], multi_class='ovo', average='macro')
    else:
        valid_auc = roc_auc_score(labels, preds)
    writer.add_scalar('valid/auc', valid_auc, epoch)
    
    preds = [round(item) for item in preds]
    if args.num_classes > 2:
        valid_recall = recall_score(labels, preds, average='macro', zero_division=np.nan)
        valid_precision = precision_score(labels, preds, average='macro', zero_division=np.nan)
    else:
        valid_recall = recall_score(labels, preds, zero_division=np.nan)
        valid_precision = precision_score(labels, preds, zero_division=np.nan)
    writer.add_scalar('valid/recall', valid_recall, epoch)
    writer.add_scalar('valid/precision', valid_precision, epoch)

    return losses.avg, accs.avg, valid_auc



if __name__ == "__main__":
    args = create_arg_parser()
    
    testHospitalName = args.hospital_test[0]
    if len(args.hospital_test) > 1:
        for i in range(len(args.hospital_test)-1):
            i += 1
            testHospitalName += '+' + args.hospital_test[i]
    
    if args.tumorCrop:
        checkTumorCrop = 'O'
    else:
        checkTumorCrop = 'X'
    
    if 'wMask' in args.model_CT:
        maskOX = 'O'
    else:
        maskOX = 'X'
    
    # CT(1) pathology(2) CI(3)
    if args.modality == ['CT', 'pathology', 'CI']:
        modality_used = '123'
        model_name = args.model_CT + '-' + args.model_pathology + '-' + args.model_CI + '(' + args.aggregator + ')'
    elif args.modality == ['CT', 'pathology']:
        modality_used = '12'
        model_name = args.model_CT + '-' + args.model_pathology + '(' + args.aggregator + ')'
    elif args.modality == ['pathology', 'CI']:
        modality_used = '23'
        model_name = args.model_pathology + '-' + args.model_CI + '(' + args.aggregator + ')'
    elif args.modality == ['CT', 'CI']:
        modality_used = '13'
        model_name = args.model_CT + '-' + args.model_CI + '(' + args.aggregator + ')'
    elif args.modality == ['CT']:
        modality_used = '1'
        model_name = args.model_CT + '(' + args.aggregator + ')'
    elif args.modality == ['pathology']:
        modality_used = '2'
        model_name = args.model_pathology + '(' + args.aggregator + ')'
    elif args.modality == ['CI']:
        modality_used = '3'
        model_name = args.model_CI + '(' + args.aggregator + ')'

    train_start_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
    
    if 'CT' in args.modality:
        save_dir = '%s/modality(%s)/stage_tr(%s)/%s/norm_[%s]/mask(%s)/crop(%s)/[%d]%s' % (testHospitalName,
                                                                                       modality_used,
                                                                                       args.cancerstageTrain,
                                                                                       model_name,
                                                                                       str(args.spacing[0])+','+str(args.spacing[1])+','+str(args.spacing[2]),
                                                                                       maskOX,
                                                                                       checkTumorCrop,
                                                                                       args.val_fold,
                                                                                       train_start_time)
    else:
        save_dir = '%s/modality(%s)/stage_tr(%s)/%s/norm_[%s]/[%d]%s'                 % (testHospitalName,
                                                                                  modality_used,
                                                                                  args.cancerstageTrain,
                                                                                  model_name,
                                                                                  str(args.spacing[0])+','+str(args.spacing[1])+','+str(args.spacing[2]),
                                                                                  args.val_fold,
                                                                                  train_start_time)
    save_dir = 'results/SavedModels/' + save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/config.txt", 'w') as f:
        for key in vars(args).keys():
            f.write(f"{key}: {vars(args)[key]}\n")
    copyfile('model/aggregator.py', save_dir+'/aggregator.py')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # os.environ['MASTER_ADDR'] = args.master_IP
    # os.environ['MASTER_PORT'] = args.master_port
    
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.enabled = True
    
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # os.environ["PYTHONHASHSEED"] = str(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, save_dir))
    else:
        main_worker(args.gpu, ngpus_per_node, args, save_dir)


    sys.exit(0)