# from tsnecuda import TSNE
from sklearn.manifold import TSNE

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

from torcheval.metrics.aggregation.auc import AUC

import shutil
import warnings
import builtins
import random
import math
import sys
from tqdm import tqdm
import pandas as pd

from dataset import ImageDataset
from model.utils import get_model

from utils import AverageMeter, calculate_accuracy, save_checkpoint, adjust_learning_rate, ProgressMeter_wID

from model.dim1 import CLIP

# from pytorch_wavelets import DWTForward, DWTInverse

#%% settings for deep learning
# from config import create_arg_parser

import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def create_arg_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--hospital',
    #                     default=['AJMC','EUMC','HUMC','PNUH','SCHMC'], type=arg_as_list, help='')
    parser.add_argument('--hospital_test',
                        default=['AJMC', 'CNUH', 'PNUH', 'EUMC', 'HUMC', 'SCHMC'], type=arg_as_list, help='')
    parser.add_argument('--kfold_num',
                        default=4, type=int, help='')
    parser.add_argument('--val_fold',
                        default=0, type=int, help='')
    
    parser.add_argument('--resampleXY', default=0, help='1: Resampling for XYZ, 0: Resampling only for Z')
    parser.add_argument('--spacing', default=[0.6869, 0.6869, 3.0], type=arg_as_list, help='')
    parser.add_argument('--tumorCrop', default=1, type=int)

    parser.add_argument('--exp_dir',
                        default='/mnt/KW/LungCancer/MIL/trained_model/224_1.0', type=str, help='')
    parser.add_argument('--type',
                        default='Biopsy+Resection', type=str)
    parser.add_argument('--test_type',
                        default='Biopsy+Resection', type=str)
    parser.add_argument('--path_data_CT',
                        default='/mai_nas/KW/Data/LungCancer/CT',
                        type=str, help='CT set')
    parser.add_argument('--path_data_pathology',
                        default='/mai_nas/KW/Data/LungCancer/Pathology_feature_CTransPath_224_1.0_sn(StainNet)',
                        type=str, help='Pathology set')
    parser.add_argument('--path_data_excel',
                        default='/mai_nas/KW/Data/LungCancer/Clinical_excel',
                        type=str, help='')
    parser.add_argument('--path_data_mask',
                        default='/mai_nas/KW/Data/LungCancer/CT/TumorMask',
                        type=str, help='')
    parser.add_argument('--clinical_features',
                        default=['sex', 'age', 'sm', 'locationcancer', 'cancerimaging', 'cancerimagingT', 'cancerimagingN', 'cancerimagingM', 'classification cancer'],
                        type=arg_as_list, help='Clinical features used for training')
    
    parser.add_argument('--modality', default=['CT', 'pathology', 'CI'], type=arg_as_list, help='CT, pathology, clinical info (CI)')
    parser.add_argument('--model_CT', default='resnetMC3_18', type=str) # resnetMC3_18, medicalNet
    parser.add_argument('--model_pathology', default='TransMIL', type=str)
    parser.add_argument('--model_CI', default='simpleFCs_v1', type=str)
    parser.add_argument('--aggregator', default='TransMIL', type=str)
    
    parser.add_argument('--prompt_len', default=0, type=int)
    
    parser.add_argument('--data_integration', default=0, type=int)
    parser.add_argument('--augmentation', default=1, type=int)
    
    parser.add_argument('--cancerstageTrain', type=str, default='1234', help='1,2,3,4,12,34,1234,etc.')
    parser.add_argument('--cancerstageTest', type=str, default='1234', help='1,2,3,4,12,34,1234,etc.')

    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrained_weights', type=str, default='DEFAULT')
    parser.add_argument('--dim', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--activationF', type=str, default='softmax')

    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Strength of weight decay regularization')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--loss', type=str, default='BCE')

    parser.add_argument('--schedule', default=[200, 500], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

    parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default='4', type=str, help='GPU Number')
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:4444')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=4444)
    parser.add_argument('--proc_idx', type=int, default=0)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--world_size', type=int, default=1)

    parser.add_argument('--iter_per_epoch', type=int, default=100)
    parser.add_argument('--val_iter_per_epoch', type=int, default=50)
    parser.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')

    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    parser.add_argument('--survival_type',
                        default='OS',
                        help='OS / RFS', type=str)
    parser.add_argument('--year',
                        default=5,
                        help='3 / 5', type=int)
    parser.add_argument('--label', default='BpRs', type=str)

    parser.add_argument('--test_pth', type=str, default=None)

    args = parser.parse_args()

    return args


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
    cudnn.benchmark = True
    
    #%%
    args.hospital_test = ['AJMC']
    test_dataset_AJMC = ImageDataset(args, mode='test')
    dataloader_test_AJMC = DataLoader(test_dataset_AJMC, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    data_AJMC = get_data(dataloader_test_AJMC, args) # [CT_zip, pathology_zip, CI_zip, label_zip]
    print("%s.. loaded!" %args.hospital_test[0])
    
    args.hospital_test = ['CNUH']
    test_dataset_CNUH = ImageDataset(args, mode='test')
    dataloader_test_CNUH = DataLoader(test_dataset_CNUH, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    data_CNUH = get_data(dataloader_test_CNUH, args)
    print("%s.. loaded!" %args.hospital_test[0])
    
    args.hospital_test = ['PNUH']
    test_dataset_PNUH = ImageDataset(args, mode='test')
    dataloader_test_PNUH = DataLoader(test_dataset_PNUH, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    data_PNUH = get_data(dataloader_test_PNUH, args)
    print("%s.. loaded!" %args.hospital_test[0])
    
    # args.hospital_test = ['EUMC', 'HUMC', 'SCHMC']
    # valid_dataset = ImageDataset(args, mode='valid')
    # dataloader_valid = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # data_valid = get_data(dataloader_valid, args) # [CT_zip, pathology_zip, CI_zip, label_zip]
    # print("Internal valid set.. loaded!")

    # args.hospital_test = ['EUMC', 'HUMC', 'SCHMC']
    # train_dataset = ImageDataset(args, mode='train')
    # dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # data_train = get_data(dataloader_train, args) # [CT_zip, pathology_zip, CI_zip, label_zip]
    # print("Train set.. loaded!")
    
    args.hospital_test = ['EUMC']
    test_dataset_EUMC = ImageDataset(args, mode='test')
    dataloader_test_EUMC = DataLoader(test_dataset_EUMC, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    data_EUMC = get_data(dataloader_test_EUMC, args)
    print("%s.. loaded!" %args.hospital_test[0])
    
    args.hospital_test = ['HUMC']
    test_dataset_HUMC = ImageDataset(args, mode='test')
    dataloader_test_HUMC = DataLoader(test_dataset_HUMC, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    data_HUMC = get_data(dataloader_test_HUMC, args)
    print("%s.. loaded!" %args.hospital_test[0])
    
    args.hospital_test = ['SCHMC']
    test_dataset_SCHMC = ImageDataset(args, mode='test')
    dataloader_test_SCHMC = DataLoader(test_dataset_SCHMC, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    data_SCHMC = get_data(dataloader_test_SCHMC, args)
    print("%s.. loaded!" %args.hospital_test[0])
    
    #%%
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#5d1371', '#8c564b']
    # colors = ['#1f77b4', '#291fb4', '#ff7f0e', '#ffbb0e', '#2ca02c', '#1bb88e', '#d62728', '#d627b3', '#5d1371', '#9467bd', '#8c564b', '#e47c57']
    
    # fig_CT = plt.figure()
    # ax_CT = fig_CT.add_subplot(111)
    # CT_tsne = TSNE().fit_transform(np.concatenate((data_AJMC[0], data_CNUH[0], data_PNUH[0], data_EUMC[0], data_HUMC[0], data_SCHMC[0]), axis=0))
    # plot_tsne(ax_CT, CT_tsne[:len(dataloader_test_AJMC),:], 'AJMC', color='#ff7f0e')
    # plot_tsne(ax_CT, CT_tsne[len(dataloader_test_AJMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH),:], 'CNUH', color='#ff7f0e')
    # plot_tsne(ax_CT, CT_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH),:], 'PNUH', color='#ff7f0e')
    # plot_tsne(ax_CT, CT_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC),:], 'EUMC', color='#291fb4')
    # plot_tsne(ax_CT, CT_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC),:], 'HUMC', color='#291fb4')
    # plot_tsne(ax_CT, CT_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC)+len(dataloader_test_SCHMC),:], 'SCHMC', color='#291fb4')
    # fig_CT.savefig("T-SNE(CT).png")
    # print('>> CT... Done!')
    
    # fig_CT = plt.figure()
    # ax_CT = fig_CT.add_subplot(111)
    # CT_tsne = TSNE().fit_transform(np.concatenate((data_train[0], data_valid[0], data_EUMC[0], data_HUMC[0], data_SCHMC[0]), axis=0))
    # plot_tsne(ax_CT, CT_tsne[:len(dataloader_train)+len(dataloader_valid)], 'Internal', color='#ff7f0e')
    # plot_tsne(ax_CT, CT_tsne[len(dataloader_train)+len(dataloader_valid):], 'External', color='#291fb4')
    # fig_CT.savefig("T-SNE(CT).png")
    # print('>> CT... Done!')
    
    # fig_pathology = plt.figure()
    # ax_pathology = fig_pathology.add_subplot(111)
    # pathology_tsne = TSNE().fit_transform(np.concatenate((data_AJMC[1], data_CNUH[1], data_PNUH[1], data_EUMC[1], data_HUMC[1], data_SCHMC[1]), axis=0))
    # plot_tsne(ax_pathology, pathology_tsne[:len(dataloader_test_AJMC),:], 'AJMC', color=colors[0])
    # # plot_tsne(ax_pathology, pathology_tsne[:len(dataloader_test_AJMC),:], 'AJMC', data_AJMC[3], color=colors[0:2])
    # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH),:], 'CNUH', color=colors[1])
    # # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH),:], 'CNUH', data_CNUH[3], color=colors[2:4])
    # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH),:], 'PNUH', color=colors[2])
    # # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH),:], 'PNUH', data_PNUH[3], color=colors[4:6])
    # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC),:], 'EUMC', color=colors[3])
    # # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC),:], 'EUMC', data_EUMC[3], color=colors[6:8])
    # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC),:], 'HUMC', color=colors[4])
    # # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC),:], 'HUMC', data_HUMC[3], color=colors[8:10])
    # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC)+len(dataloader_test_SCHMC),:], 'SCHMC', color=colors[5])
    # # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC)+len(dataloader_test_SCHMC),:], 'SCHMC', data_SCHMC[3], color=colors[10:12])
    # fig_pathology.savefig("T-SNE(pathology).png")
    # print('>> Pathology... Done!')
    
    # fig_pathology = plt.figure()
    # ax_pathology = fig_pathology.add_subplot(111)
    # pathology_tsne = TSNE().fit_transform(np.concatenate((data_train[1], data_valid[1], data_EUMC[1], data_HUMC[1], data_SCHMC[1]), axis=0))
    # plot_tsne(ax_pathology, pathology_tsne[:len(dataloader_train)+len(dataloader_valid)], 'Internal', color='#ff7f0e')
    # plot_tsne(ax_pathology, pathology_tsne[len(dataloader_train)+len(dataloader_valid):], 'External', color='#291fb4')
    # fig_pathology.savefig("T-SNE(pathology).png")
    # print('>> Pathology... Done!')
    
    fig_pathology = plt.figure()
    ax_pathology = fig_pathology.add_subplot(111)
    pathology_tsne = TSNE().fit_transform(np.concatenate((data_AJMC[1], data_CNUH[1], data_PNUH[1], data_EUMC[1], data_HUMC[1], data_SCHMC[1]), axis=0))
    plot_tsne(ax_pathology, pathology_tsne[:len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)], 'Internal', color='#ff7f0e')
    plot_tsne(ax_pathology, pathology_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH):], 'External', color='#291fb4')
    fig_pathology.savefig("T-SNE(pathology)_v2.png")
    print('>> Pathology... Done!')
    
    # fig_CI = plt.figure()
    # ax_CI = fig_CI.add_subplot(111)
    # CI_tsne = TSNE().fit_transform(np.concatenate((data_AJMC[2], data_CNUH[2], data_PNUH[2], data_EUMC[2], data_HUMC[2], data_SCHMC[2]), axis=0))
    # plot_tsne(ax_CI, CI_tsne[:len(dataloader_test_AJMC),:], 'AJMC', color=colors[0])
    # plot_tsne(ax_CI, CI_tsne[len(dataloader_test_AJMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH),:], 'CNUH', color=colors[1])
    # plot_tsne(ax_CI, CI_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH),:], 'PNUH', color=colors[2])
    # plot_tsne(ax_CI, CI_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC),:], 'EUMC', color=colors[3])
    # plot_tsne(ax_CI, CI_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC),:], 'HUMC', color=colors[4])
    # plot_tsne(ax_CI, CI_tsne[len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC):len(dataloader_test_AJMC)+len(dataloader_test_CNUH)+len(dataloader_test_PNUH)+len(dataloader_test_EUMC)+len(dataloader_test_HUMC)+len(dataloader_test_SCHMC),:], 'SCHMC', color=colors[5])
    # fig_CI.savefig("T-SNE(CI).png")
    # print('>> Clinical Information... Done!')
    
    # fig_CI = plt.figure()
    # ax_CI = fig_CI.add_subplot(111)
    # CI_tsne = TSNE().fit_transform(np.concatenate((data_train[2], data_valid[2], data_EUMC[2], data_HUMC[2], data_SCHMC[2]), axis=0))
    # plot_tsne(ax_CI, CI_tsne[:len(dataloader_train)+len(dataloader_valid)], 'Internal', color='#ff7f0e')
    # plot_tsne(ax_CI, CI_tsne[len(dataloader_train)+len(dataloader_valid):], 'External', color='#291fb4')
    # fig_CI.savefig("T-SNE(CI).png")
    # print('>> Clinical Information... Done!')


def get_data(dataloader_test, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    accs = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter_wID(
        len(dataloader_test),
        [batch_time, data_time, losses, accs],
        prefix="Test: ",
    )
    
    # AUC_metric = AUC()
    DF = pd.DataFrame()
    
    # if 'CT' in args.modality:
    CT_zip = np.zeros((len(dataloader_test),160*512*512))
    # if 'pathology' in args.modality:
    # pathology_zip = np.zeros((len(dataloader_test),14880*768))
    pathology_zip = np.zeros((len(dataloader_test),15592*768))
    # if 'CI' in args.modality:
    CI_zip = np.zeros((len(dataloader_test),512))
    
    label_zip = np.zeros((len(dataloader_test),1))
    
    with torch.no_grad():
        clinic_extractor = CLIP(args).cuda()
        
        for i, test_data_dict in tqdm(enumerate(dataloader_test)):
            if args.gpu is not None:
                if ('CT' in args.modality) & ('pathology' in args.modality) & ('CI' in args.modality):
                    test_input_CT = test_data_dict['input_CT'].cuda(non_blocking=True)
                    test_input_pathology = test_data_dict['input_pathology'].cuda(non_blocking=True)
                    test_input_CI = test_data_dict['input_CI'].cuda(non_blocking=True)
                elif ('CT' in args.modality) & ('pathology' in args.modality):
                    test_input_CT = test_data_dict['input_CT'].cuda(non_blocking=True)
                    test_input_pathology = test_data_dict['input_pathology'].cuda(non_blocking=True)
                elif ('CT' in args.modality) & ('CI' in args.modality):
                    test_input_CT = test_data_dict['input_CT'].cuda(non_blocking=True)
                    test_input_CI = test_data_dict['input_CI'].cuda(non_blocking=True)
                elif ('pathology' in args.modality) & ('CI' in args.modality):
                    test_input_pathology = test_data_dict['input_pathology'].cuda(non_blocking=True)
                    test_input_CI = test_data_dict['input_CI'].cuda(non_blocking=True)
                elif ('CT' in args.modality):
                    test_input_CT = test_data_dict['input_CT'].cuda(non_blocking=True)
                elif ('pathology' in args.modality):
                    test_input_pathology = test_data_dict['input_pathology'].cuda(non_blocking=True)
                elif ('CI' in args.modality):
                    test_input_CI = test_data_dict['input_CI'].cuda(non_blocking=True)
                    
                if 'wMask' in args.model_CT:
                    test_mask = test_data_dict['mask'].cuda(non_blocking=True)
                
                if 'pathology' in args.modality:
                    if args.model_pathology == 'ABMIL_v2':
                        test_info_BpRc = test_data_dict['BpRc_class'].float().cuda(non_blocking=True)
                
                test_label = test_data_dict['label'].float().cuda(non_blocking=True)
                test_ID = test_data_dict['ID'][0]
                
                label_zip[i,0] = test_label[0,1].detach().cpu().numpy()
            
            if 'CT' in args.modality:
                CT_zip[i,:] = test_input_CT.detach().cpu().numpy().flatten()
            if 'pathology' in args.modality:
                pathology_zip[i,:test_input_pathology.shape[1]*test_input_pathology.shape[2]] = test_input_pathology.detach().cpu().numpy().flatten()
            if 'CI' in args.modality:
                CI_zip[i,:] = clinic_extractor(test_input_CI).detach().cpu().numpy()
    
    return CT_zip, pathology_zip, CI_zip, label_zip
    # if ('CT' in args.modality) & ('pathology' in args.modality) & ('CI' in args.modality):
    #     return CT_tsne, pathology_tsne, CI_tsne
    # elif ('CT' in args.modality) & ('pathology' in args.modality):
    #     return CT_tsne, pathology_tsne
    # elif ('CT' in args.modality) & ('CI' in args.modality):
    #     return CT_tsne, CI_tsne
    # elif ('pathology' in args.modality) & ('CI' in args.modality):
    #     return pathology_tsne, CI_tsne
    # elif ('CT' in args.modality):
    #     return CT_tsne
    # elif ('pathology' in args.modality):
    #     return pathology_tsne
    # elif ('CI' in args.modality):
    #     return CI_tsne


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    smooth = 1e-10
    return (starts_from_zero + smooth) / (value_range + smooth)


def plot_tsne(ax, tsne, hospital, color):
    tsne_x = scale_to_01_range(tsne[:, 0])
    tsne_y = scale_to_01_range(tsne[:, 1])
    
    ax.scatter(tsne_x, tsne_y, c=color, label=hospital)
    
    ax.legend(loc='best')
    
# def plot_tsne(ax, tsne, hospital, label, color):
#     tsne_x = scale_to_01_range(tsne[:, 0])
#     tsne_y = scale_to_01_range(tsne[:, 1])
    
#     indices_0 = [i for i, l in enumerate(label) if l == 0]
#     indices_1 = [i for i, l in enumerate(label) if l == 1]
        
#     tsne_x0 = np.take(tsne_x, indices_0)
#     tsne_y0 = np.take(tsne_y, indices_0)
#     # ax.scatter(tsne_x0, tsne_y0, c=color[0], label='%s-Resection'%hospital)
#     ax.scatter(tsne_x0, tsne_y0, c=color[0])
    
#     tsne_x1 = np.take(tsne_x, indices_1)
#     tsne_y1 = np.take(tsne_y, indices_1)
#     # ax.scatter(tsne_x1, tsne_y1, c=color[1], label='%s-Biopsy'%hospital)
#     ax.scatter(tsne_x1, tsne_y1, c=color[1])
    
#     ax.legend(loc='best')


from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt

def ROC_curve_plot(label, score, save_path):
    fpr, tpr, thresholds = roc_curve(label, score)
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
    
    return auc



if __name__ == "__main__":
    args = create_arg_parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
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