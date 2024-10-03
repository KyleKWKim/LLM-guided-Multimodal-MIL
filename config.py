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
                        default=['EUMC', 'HUMC', 'SCHMC'], type=arg_as_list, help='')
    parser.add_argument('--kfold_num',
                        default=4, type=int, help='')
    parser.add_argument('--val_fold',
                        default=0, type=int, help='')
    
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--best_thres', type=float, default=0.5)

    parser.add_argument('--resampleXY', default=1, help='1: Resampling for XYZ, 0: Resampling only for Z')
    parser.add_argument('--spacing', default=[0.6869, 0.6869, 3.0], type=arg_as_list, help='')
    parser.add_argument('--tumorCrop', default=0, type=int)

    # parser.add_argument('--exp_dir',
    #                     default='/mnt/KW/LungCancer/MIL/trained_model/224_1.0', type=str, help='')
    parser.add_argument('--type',
                        default='Biopsy+Resection', type=str)
    parser.add_argument('--test_type',
                        default='Biopsy+Resection', type=str)
    parser.add_argument('--path_data_CT',
                        default='/mai_nas/KW/Data/LungCancer/CT',
                        type=str, help='CT set')
    parser.add_argument('--path_feature_CT',
                        default='/mai_nas/KW/Data/LungCancer/CT/X(2.0)Y(2.0)Z(2.5)/SwinUNETR_feature',
                        type=str, help='Feature of CT via SwinUNETR')
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
                        default=['sex', 'age', 'sm', 'locationcancer', 'cancerimaging', 'cancerimagingT', 'cancerimagingN', 'cancerimagingM', 'classification_cancer'],
                        type=arg_as_list, help='Clinical features used for training')
    parser.add_argument('--pathology_info_version', default='v1', type=str)
    
    parser.add_argument('--modality', default=['CT', 'pathology'], type=arg_as_list, help='CT, pathology, clinical info (CI)')
    parser.add_argument('--alignment_base', default='none', type=str, help='CT, pathology, CI, none')
    parser.add_argument('--model_CT', default='resnetMC3_18', type=str) # resnetMC3_18, medicalNet
    parser.add_argument('--model_pathology', default='TransMIL', type=str)
    parser.add_argument('--CI_input_version', default='v1', type=str, help='v1,v2,text')
    parser.add_argument('--CI_prompt_version', default='single', type=str, help='single, devided')
    parser.add_argument('--model_CI', default='simpleFCs_v1', type=str, help='simpleFCs_v1, CLIP, ...')
    parser.add_argument('--aggregator', default='TransMIL', type=str)
    
    parser.add_argument('--learnablePrompt', default=1, type=int)
    parser.add_argument('--n_ctx', default=8, type=int)
    parser.add_argument('--n_prompts', default=2, type=int)
    # parser.add_argument('--text_format', default="37 year old male nonsmoker lung cancer patient, stage 3, T1 N2 M0, location right superior lobe, type adenocarcinoma", type=str)
    parser.add_argument('--prompt_len', default=0, type=int)
    
    parser.add_argument('--data_integration', default=0, type=int)
    parser.add_argument('--augmentation', default=1, type=int)
    
    parser.add_argument('--cancerstageTrain', type=str, default='1234', help='1,2,3,4,12,34,1234,etc.')
    parser.add_argument('--cancerstageTest', type=str, default='1234', help='1,2,3,4,12,34,1234,etc.')
    
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrained_weights', type=str, default='DEFAULT')
    parser.add_argument('--dim', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--activationF', type=str, default='sigmoid')

    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--pretrainedExt_CT', type=int, default=1)
    parser.add_argument('--pretrainedExt_CT_pth', type=str, default='/mnt/KW/LungCancer/Multimodality/results/SavedModels/modality(1)/stage_tr(1234)/norm_[2.0,2.0,2.5]/mask(X)/crop(X)/2023-12-18-15:44:18/checkpoint_0901.pth.tar')
    parser.add_argument('--pretrainedExt_pathology', type=int, default=1)
    # parser.add_argument('--pretrainedExt_pathology_pth', type=str, default='/mnt/KW/LungCancer/Multimodality/results/SavedModels/EUMC+HUMC+SCHMC/modality(2)/stage_tr(1234)/ABMIL_v2/norm_[2.0,2.0,2.5]/2024-01-30-14:07:36/checkpoint_0009.pth.tar')
    parser.add_argument('--pretrainedExt_pathology_pth', type=str, default='/mnt/KW/LungCancer/Multimodality/results/SavedModels/EUMC+HUMC+SCHMC/modality(2)/stage_tr(1234)/ABMIL_v2/norm_[2.0,2.0,2.5]/2024-01-30-14:07:36/checkpoint_0009.pth.tar')
    parser.add_argument('--pretrainedExt_CI', type=int, default=1)
    parser.add_argument('--pretrainedExt_CI_pth', type=str, default='/mnt/KW/LungCancer/Multimodality/results/SavedModels/EUMC+HUMC+SCHMC/modality(3)/stage_tr(1234)/simpleFCs_v2/norm_[2.0,2.0,2.5]/2024-01-31-07:27:00/checkpoint_0473.pth.tar')
    parser.add_argument('--pretrainedExt_freeze', type=int, default=1)

    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Strength of weight decay regularization')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate') # CI only : 0.0001 | pathology only : 0.0001
    parser.add_argument('--loss', type=str, default='BCE+CLIP', help='BCE, CS, BCE+CS')
    parser.add_argument('--loss_point', type=str, default='CT-Pth-Last')

    parser.add_argument('--schedule', default=[500], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

    parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default='4', type=str, help='GPU Number')
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:4444')
    # parser.add_argument('--dist_url', type=str, default='tcp://165.132.56.84:4444')
    parser.add_argument('--master_IP', type=str, default='165.132.56.84', help='localhost, 165.132.56.84 등으로 지정 가능')
    parser.add_argument('--master_port', type=str, default='4444')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--port', type=int, default=4444)
    parser.add_argument('--proc_idx', type=int, default=0)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--world_size', type=int, default=1, help='# for # nodes')

    parser.add_argument('--iter_per_epoch', type=int, default=100)
    parser.add_argument('--val_iter_per_epoch', type=int, default=50)
    parser.add_argument('--batch_size', default=8, type=int, help='Mini batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of jobs')
    
    parser.add_argument('--save_best', action='store_true')

    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

    parser.add_argument('--survival_type',
                        default='OS',
                        help='OS / RFS', type=str)
    parser.add_argument('--year',
                        default=5,
                        help='3 / 5', type=int)
    parser.add_argument('--label', default='5yOS', type=str) # 5yOS, Tstage, Classification Cancer, Biopsy/Resection
    
    parser.add_argument('--test_pth', type=str, default=None)
    
    parser.add_argument('--watch_ID', type=str, default='A000000')
    parser.add_argument('--map_type', type=str, default='saliencyMap', help='saliencyMap, gradCAM')

    args = parser.parse_args()

    return args