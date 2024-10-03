import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
from scipy import ndimage
import cv2
import os
from multiprocessing.pool import Pool
import pandas as pd
import copy

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.model_selection import KFold
from glob import glob
# import nrrd

from monai import transforms
from monai.transforms import *

import pydicom
import clip


pd.set_option('mode.chained_assignment',  None)


class ImageDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.transform = self.make_transform()
        
        # Data paths of CT / Pathology / Clinical info.
        self.CT_path = self.args.path_data_CT + '/X('+str(self.args.spacing[0])+')Y('+str(self.args.spacing[1])+')Z('+str(self.args.spacing[2])+')'
        self.pathology_path = self.args.path_data_pathology
        self.clinical_features = self.args.clinical_features

        if (self.args.tumorCrop) or ('wMask' in self.args.model_CT):
            self.tumorMask_path = self.args.path_data_mask + '/inference_imagesTs_3d_cascade_fullres_ALL'
            self.fillHoles = FillHoles()
            self.largestComponent = KeepLargestConnectedComponent()
        
        # Select list of samples
        # df_overall = pd.read_excel(self.args.path_data_excel + '/Lung_Overall.xlsx')
        df_overall = pd.read_excel(self.args.path_data_excel + '/Lung_Overall2.xlsx')
        # if 'CT' in self.args.modality:
        df_overall = df_overall.loc[~df_overall['CT_before1'].isna()]
        df_overall = df_overall.loc[df_overall['CT ID mismatch'].isin([0])]
        # if 'pathology' in self.args.modality:
        df_overall = df_overall.loc[df_overall['pathologyimage'].isin(['Biopsy', 'Resection'])]
        
        df_selected = self.data_selection_wLabel(df_overall)
        df_selected = self.data_selection(df_selected)
        
        # if 'CT' in self.args.modality:
        #     df_selected['CT_dir'] = self.CT_path + '/' + df_selected['hospital'] + '/' + df_selected['patientid'] + '.nii.gz'
        # if 'pathology' in self.args.modality:
        #     df_selected['pathology_dir'] = self.pathology_path + '/' + df_selected['hospital'] + '/' + df_selected['pathologyimage'] + '/' + df_selected['patientid'] + '.npy'

        if self.mode == 'test':
            self.hospital = self.args.hospital_test
        else:
            self.hospital = ['AJMC','EUMC','CNUH','HUMC','PNUH','SCHMC']
            # self.hospital = ['AJMC', 'CNUH', 'PNUH']
            for H in range(len(self.args.hospital_test)):
                self.hospital.remove(self.args.hospital_test[H])
        
        df_selected = df_selected.loc[df_selected['hospital'].isin(self.hospital)]

        if self.mode == 'test':
            self.df = df_selected
        else:
            self.val_fold = self.args.val_fold
            kf = KFold(n_splits=self.args.kfold_num, shuffle=True, random_state=42)

            for i, (foldT, foldV) in enumerate(kf.split(df_selected)):
                if i == self.args.val_fold:
                    if self.mode == 'valid':
                        self.df = df_selected.iloc[foldV]
                    elif self.mode == 'train':
                        self.df = df_selected.iloc[foldT]

        removed = [# CT영상이 A10065는 90도, A10094는 180도 회전되어 있음.
                   'A10065', 'A10094', # AJMC
                   # CT영상의 z-slice 수가 너무 적음
                   'A11512', 'A12867', # AJMC
                   'A10237', 'A110004', 'A110027', 'A110541', # CNUH
                   'A40186', 'A40211', 'A40222', 'A40235', 'A40251', 'A40261', 'A40273', 'A40274', 'A40280', 'A40281',
                   'A40282', 'A40301', 'A40308', 'A40322', 'A40326', 'A40343', 'A40348', 'A40359', 'A40368', 'A40374',
                   'A40375', 'A40379', 'A40393', 'A40430', 'A40453', 'A40461', 'A40472', 'A40478', 'A40490', 'A40494',
                   'A40495', 'A40505', 'A40541', 'A40629', 'A40637', 'A40856', 'A40860', 'A40865', 'A40884', 'A40914',
                   'A40932', 'A40941', 'A40947', 'A40949', 'A40958', 'A40966', 'A40968', 'A40988', 'A40991', 'A40992',
                   'A40993', 'A41011', 'A41056', 'A41060', 'A41508', 'A41510', 'A41513', 'A41521', 'A41547', 'A41548',
                   'A41557', 'A41592', 'A41599', 'A41601', 'A41614', 'A41619', 'A41622', 'A41637', 'A41653', 'A41690',
                   'A41704', 'A41710', 'A41712', # PNUH
                   'A60253', 'A60374', # SCHMC
                   'A93650', 'A96982', # EUMC
                   'A131302', 'A131341', # HUMC
                   # CT영상 crop이 심하게 됨.
                   'A70312', # SCHMC
                   'A90169', 'A91031', 'A93350', 'A93761', 'A96937' # EUMC
        ]
        # removed = ['A11626', 'A29822', 'A98930', 'A110493', 'A110572', 'A112004', 'A130066', 'A130920', 'A131127', 'A40636', 'A41072', 'A41597', 'A41768']
        for ID in removed:
            self.df.drop(self.df[self.df['patientid']==ID].index, inplace=True)
        
        
        # cancer stage sorting
        if self.mode == 'train':
            cancerstage = self.args.cancerstageTrain
        else:
            cancerstage = self.args.cancerstageTest
        
        if cancerstage == '1':
            self.df = self.df[self.df['cancerimaging'].isin([1,'1','1a','1b','1c'])]
        elif cancerstage == '2':
            self.df = self.df[self.df['cancerimaging'].isin([2,'2','2a','2b','2c'])]
        elif cancerstage == '3':
            self.df = self.df[self.df['cancerimaging'].isin([3,'3','3a','3b','3c'])]
        elif cancerstage == '4':
            self.df = self.df[self.df['cancerimaging'].isin([4,'4','4a','4b','4c'])]
        elif cancerstage == '12':
            self.df = self.df[self.df['cancerimaging'].isin([1,'1','1a','1b','1c',2,'2','2a','2b','2c'])]
        elif cancerstage == '34':
            self.df = self.df[self.df['cancerimaging'].isin([3,'3','3a','3b','3c',4,'4','4a','4b','4c'])]
        elif cancerstage == '1234':
            self.df = self.df
        


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        if 'wMask' in self.args.model_CT:
            if ('CT' in self.args.modality) & ('pathology' in self.args.modality):
                input_CT, input_pathology, input_CI, label, mask = self.getdata_from_df(self.df, idx % len(self.df))
            elif ('CT' in self.args.modality):
                input_CT, input_CI, label, mask = self.getdata_from_df(self.df, idx % len(self.df))
            elif ('pathology' in self.args.modality):
                input_pathology, input_CI, label, mask = self.getdata_from_df(self.df, idx % len(self.df))
        else:
            if ('CT' in self.args.modality) & ('pathology' in self.args.modality):
                input_CT, input_pathology, input_CI, label = self.getdata_from_df(self.df, idx % len(self.df))
            elif ('CT' in self.args.modality):
                input_CT, input_CI, label = self.getdata_from_df(self.df, idx % len(self.df))
            elif ('pathology' in self.args.modality):
                if self.args.model_pathology == 'ABMIL_v2':
                    input_pathology, input_CI, label, BpRc_class = self.getdata_from_df(self.df, idx % (len(self.df)))
                else:
                    input_pathology, input_CI, label = self.getdata_from_df(self.df, idx % len(self.df))
            elif ('CI' in self.args.modality):
                input_CI, label = self.getdata_from_df(self.df, idx % len(self.df))
        
        ID = self.df.iloc[idx % len(self.df)]['patientid']
        
        if 'wMask' in self.args.model_CT:
            if ('CT' in self.args.modality) & ('pathology' in self.args.modality):
                data_dict = {'input_CT' : input_CT.unsqueeze(0), 'input_pathology' : input_pathology, 'input_CI' : input_CI, 'label' : label, 'mask' : mask.unsqueeze(0), 'ID' : ID}
            elif ('CT' in self.args.modality):
                data_dict = {'input_CT' : input_CT.unsqueeze(0), 'input_CI' : input_CI, 'label' : label, 'mask' : mask.unsqueeze(0), 'ID' : ID}
            elif ('pathology' in self.args.modality):
                data_dict = {'input_pathology' : input_pathology, 'input_CI' : input_CI, 'label' : label, 'mask' : mask.unsqueeze(0), 'ID' : ID}
        else:
            if ('CT' in self.args.modality) & ('pathology' in self.args.modality):
                data_dict = {'input_CT' : input_CT.unsqueeze(0), 'input_pathology' : input_pathology, 'input_CI' : input_CI, 'label' : label, 'ID' : ID}
            elif ('CT' in self.args.modality):
                data_dict = {'input_CT' : input_CT.unsqueeze(0), 'input_CI' : input_CI, 'label' : label, 'ID' : ID}
            elif ('pathology' in self.args.modality):
                if self.args.model_pathology == 'ABMIL_v2':
                    data_dict = {'input_pathology' : input_pathology, 'input_CI' : input_CI, 'label' : label, 'BpRc_class' : BpRc_class, 'ID' : ID}
                else:
                    data_dict = {'input_pathology' : input_pathology, 'input_CI' : input_CI, 'label' : label, 'ID' : ID}
            elif ('CI' in self.args.modality):
                data_dict = {'input_CI' : input_CI, 'label' : label, 'ID' : ID}
        
        data_dict = self.transform(data_dict)

        return data_dict


    def make_transform(self):
        keys_totensor = ["label"]
        if 'wMask' in self.args.model_CT:
            keys_totensor.append("mask")
        if 'CT' in self.args.modality:
            keys_totensor.append("input_CT")
        if 'pathology' in self.args.modality:
            keys_totensor.append("input_pathology")
        
        transform = ToTensord(keys=keys_totensor)
        if self.mode == 'train':
            if self.args.augmentation:
                if 'CT' in self.args.modality:
                    if 'wMask' in self.args.model_CT:
                        transform = transforms.Compose([
                                                        # CenterSpatialCropd(keys=["input_CT", "mask"], roi_size=(-1,448,448)),
                                                        RandAffined(keys=["input_CT", "mask"], mode="nearest", prob=0.2,
                                                                    rotate_range=(np.pi/18, np.pi/18, np.pi/18), scale_range=(0.0, 0.0, 0.0), padding_mode="border"),
                                                        RandGaussianNoised(keys=["input_CT"], mean=0, std=0.05, prob=0.1),
                                                        # RandGaussianSmoothd(keys=["input_CT", "mask"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.2),
                                                        # RandScaleIntensityd(keys=["input_CT", "mask"], factors=0.25, prob=0.15),
                                                        # RandAdjustContrastd(keys=["input_CT", "mask"], gamma=(0.75, 1.25), prob=0.15),
                                                        RandFlipd(keys=["input_CT", "mask"], prob=0.2, spatial_axis=(0,1,2)),
                                                        transform
                                                        ])
                    else:
                        transform = transforms.Compose([
                                                        # CenterSpatialCropd(keys=["input_CT", "mask"], roi_size=(-1,448,448)),
                                                        RandAffined(keys=["input_CT"], mode="nearest", prob=0.2,
                                                                    rotate_range=(np.pi/18, np.pi/18, np.pi/18), scale_range=(0.0, 0.0, 0.0), padding_mode="border"),
                                                        RandGaussianNoised(keys=["input_CT"], mean=0, std=0.05, prob=0.1),
                                                        # RandGaussianSmoothd(keys=["input_CT", "mask"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.2),
                                                        # RandScaleIntensityd(keys=["input_CT", "mask"], factors=0.25, prob=0.15),
                                                        # RandAdjustContrastd(keys=["input_CT", "mask"], gamma=(0.75, 1.25), prob=0.15),
                                                        RandFlipd(keys=["input_CT"], prob=0.2, spatial_axis=(0,1,2)),
                                                        transform
                                                        ])

        return transform


    def getdata_from_df(self, df, idx):
        df_idx = df.iloc[idx]
        if self.args.label == '5yOS':
            label = df_idx['label']
        elif self.args.label == 'BpRs':
            label_ = df_idx['pathologyimage']
            if label_ == 'Biopsy':
                label = 1
            elif label_ == 'Resection':
                label = 0
        elif self.args.label == 'Tstage':
            label = df_idx['label_T']
        elif self.args.label == 'classification_cancer':
            label = df_idx['classification cancer'] - 1
        elif self.args.label == 'locationcancer':
            label = df_idx['locationcancer'] - 1
            
        # label = torch.tensor(label).unsqueeze(-1)
        label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.args.num_classes)
        
        #%%
        if 'CT' in self.args.modality:
            if 'SwinUNETR' in self.args.model_CT:
                feature_pth = self.args.path_feature_CT + '/' + df_idx['hospital'] + '/' + df_idx['patientid'] + '.npy'
                input_CT = np.load(feature_pth)
                input_CT = torch.from_numpy(input_CT).float()
            else:
                metadata_pth = self.args.path_data_CT + '/' + df_idx['hospital'] + '/' + df_idx['patientid'] + '/' + df_idx['CT_before1'][1:-3] + '/' + df_idx['CT_before1'][-2] \
                            + '/LUNG_' + df_idx['hospital'] + '_' + df_idx['patientid'] + '_CT_' + df_idx['CT_before1'][1] + '_' + df_idx['CT_before1'][-2] + '_0003.dcm'
                metadata = pydicom.read_file(metadata_pth)
                img_itk = sitk.ReadImage(self.CT_path + '/' + df_idx['hospital'] + '/' + df_idx['patientid'] + '.nii.gz')
                img = sitk.GetArrayFromImage(img_itk).squeeze()

                # if (self.args.tumorCrop) or ('wMask' in self.args.model_CT):
                #     maskF, headerF = nrrd.read(self.tumorMask_path + '/CT_' + df_idx['patientid'] + '.nrrd')
                #     maskL, headerL = nrrd.read(self.tumorMask_path + '/prediction_3d_lowres/CT_' + df_idx['patientid'] + '.nrrd')
                #     mask = maskF + maskL
                #     mask[mask>=1] = 1
                #     mask[mask<0]  = 0

                #     mask = self.largestComponent(self.fillHoles(mask))
                #     mask = mask.permute(2,0,1)
                #     sitk_mask = sitk.GetImageFromArray(mask)
                #     sitk_mask.SetSpacing([1.0, 1.0, 1.0])
                #     mask_size = sitk_mask.GetSize()
                #     out_spacing = self.args.spacing
                #     new_size = [int(orgsz*(orgspc/outspc)) for orgsz, orgspc, outspc in zip(mask_size, [1.0,1.0,1.0], out_spacing)]

                #     resample_mask = sitk.ResampleImageFilter()
                #     resample_mask.SetOutputSpacing(out_spacing)
                #     resample_mask.SetSize(new_size)
                #     resample_mask.SetTransform(sitk.Transform())
                #     resample_mask.SetDefaultPixelValue(sitk_mask.GetPixelIDValue())
                #     sitk_mask_resampled = resample_mask.Execute(sitk_mask)

                #     mask = sitk.GetArrayFromImage(sitk_mask_resampled).squeeze()

                #     if img.shape != mask.shape:
                #         if img.shape[0] < mask.shape[0]:
                #             mask = mask[:img.shape[0],:,:]
                #         else:
                #             diff = img.shape[0] - mask.shape[0]
                #             mask = np.pad(mask, ((int(diff/2),diff-int(diff/2)),(0,0),(0,0)), mode='constant')

                #         if img.shape[1] < mask.shape[1]:
                #             mask = mask[:,:img.shape[1],:img.shape[2]]
                #         else:
                #             diff = img.shape[1] - mask.shape[1]
                #             mask = np.pad(mask, ((0,0),(int(diff/2),diff-int(diff/2)),(int(diff/2),diff-int(diff/2))), mode='constant')

                C, H, W = img.shape
                if self.args.spacing[0] == 2.0:
                    H_ = 224
                    W_ = 224
                elif self.args.spacing[0] == 0.6869:
                    H_ = 512
                    W_ = 512
                C_ = 160
                
                if self.args.dim == '3d':
                    # if C >= 160:
                    #     C_diff = C - 160
                    #     img = img[int(C_diff/2):int(C_diff/2)+160,:,:]
                    # else:
                    #     C_diff = 160 - C
                    #     img = np.pad(img, ((int(C_diff/2),C_diff-int(C_diff/2)),(0,0),(0,0)), mode='constant')

                    if C >= C_:
                        img = img[:C_,:,:]
                        img = self.preprocessing_with_metadata(img, metadata)
                        # if (self.args.tumorCrop) or ('wMask' in self.args.model_CT):
                        #     mask = mask[:C_,:,:]
                    else:
                        C_diff = C_ - C
                        img = self.preprocessing_with_metadata(img, metadata)
                        img = np.pad(img, ((int(C_diff/2),C_diff-int(C_diff/2)),(0,0),(0,0)), mode='constant')
                        # if (self.args.tumorCrop) or ('wMask' in self.args.model_CT):
                        #     mask = np.pad(mask, ((int(C_diff/2),C_diff-int(C_diff/2)),(0,0),(0,0)), mode='constant')

                    if H >= H_:
                        H_diff = H - H_
                        img = img[:,int(H_diff/2):int(H_diff/2)+H_,int(H_diff/2):int(H_diff/2)+H_]
                        # if (self.args.tumorCrop) or ('wMask' in self.args.model_CT):
                        #     mask = mask[:,int(H_diff/2):int(H_diff/2)+H_,int(H_diff/2):int(H_diff/2)+H_]
                    else:
                        H_diff = H_ - H
                        img = np.pad(img, ((0,0),(int(H_diff/2),H_diff-int(H_diff/2)),(int(H_diff/2),H_diff-int(H_diff/2))), mode='constant')
                        # if (self.args.tumorCrop) or ('wMask' in self.args.model_CT):
                        #     mask = np.pad(mask, ((0,0),(int(H_diff/2),H_diff-int(H_diff/2)),(int(H_diff/2),H_diff-int(H_diff/2))), mode='constant')
                    
                    if self.args.tumorCrop:
                        mask2crop = np.zeros_like(mask)
                        mask_coordx, mask_coordy, mask_coordz = np.where(mask==1)
                        if len(mask_coordx) == 0:
                            mask_center = [int(mask.shape[0]/2), int(mask.shape[1]/2), int(mask.shape[2]/2)]
                        else:
                            mask_center = [int((c.max()-c.min())/2) for c in [mask_coordx, mask_coordy, mask_coordz]]
                        mask2crop[max(0,mask_center[0]-50):min(mask.shape[0],mask_center[0]+50),
                                max(0,mask_center[1]-50):min(mask.shape[1],mask_center[1]+50),
                                max(0,mask_center[2]-50):min(mask.shape[2],mask_center[2]+50)] = 1
                        # mask2crop[max(0,min(mask_coordx)-20):min(mask.shape[0],max(mask_coordx)+20),
                        #           max(0,min(mask_coordy)-20):min(mask.shape[1],max(mask_coordy)+20),
                        #           max(0,min(mask_coordz)-20):min(mask.shape[2],max(mask_coordz)+20)] = 1

                # if self.args.tumorCrop:
                #     mask_coordx, mask_coordy, mask_coordz = np.where(mask==1)
                #     mask_center = [(c.max()-c.min())/2 for c in [mask_coordx, mask_coordy, mask_coordz]]
                
                input_CT = torch.from_numpy(img).float()
                if self.args.tumorCrop:
                    mask2crop = torch.from_numpy(mask2crop).float()
                    # mask2crop = torch.tensor(mask2crop)
                    input_CT = input_CT * mask2crop
        
        #%%
        if 'pathology' in self.args.modality:
            feature_real = np.load(self.pathology_path + '/' + df_idx['hospital'] + '/' + df_idx['pathologyimage'] + '/' + df_idx['patientid'] + '.npy')
            
            # idx_list = np.arange(feature_real.shape[0])
            # np.random.shuffle(idx_list)
            # feature_real = feature_real[idx_list,:]
            
            n, _ = feature_real.shape
            if self.mode == 'train':
                if self.args.augmentation:
                    if df_idx['pathologyimage'] == 'Biopsy':
                        # feature_real = feature_real[sorted(random.sample(range(n),int(n*(1-(n/N)*(1/3))))),:]
                        feature_real = feature_real[sorted(random.sample(range(n),int(n*(1-0.1)))),:]
                    elif df_idx['pathologyimage'] == 'Resection':
                        # feature_real = feature_real[sorted(random.sample(range(n),int(n*(1-(n/N)*(2/3))))),:]
                        feature_real = feature_real[sorted(random.sample(range(n),int(n*(1-0.2)))),:]
            
            if self.args.batch_size == 1:
                feature = feature_real
            else:
                if self.args.path_data_pathology[-3:] == '_sn':
                    feature = np.zeros((14880,768))
                else:
                    feature = np.zeros((15592,768))
                N, _ = feature.shape
                feature[:feature_real.shape[0],:] = feature_real
            
            input_pathology = torch.from_numpy(feature).float()
            
            if self.args.model_pathology == 'ABMIL_v2':
                BpRc = df_idx['pathologyimage']
                if BpRc == 'Biopsy':
                    BpRc_class = torch.tensor(0).to(torch.int64).unsqueeze(-1)
                elif BpRc == 'Resection':
                    BpRc_class = torch.tensor(1).to(torch.int64).unsqueeze(-1)
        
        #%%
        # [sex, age, sm, locationcancer, cancerimaging, cancerimagingT, cancerimagingN, cancerimagingM, classification_cancer]
        # sex : male(0), female(1)
        # age : 00 year old
        # sm : nonsmoker(0), smoker(1)
        # ----------------------------- ex) 32 year old male nonsmoker
        # locationcaner : right superior lobe(1), right middle lobe(2), right inferior lobe(3), left superior lobe(4), left inferior lobe(5)
        # cancerimaging : stage 1(2,3,4,5) of
        # cancerimagingT : T1(2,3,4)
        # cancerimagingN : N1(2,3,4)
        # cancerimagingM : M0
        # ----------------------------- ex) lung cancer stage 3, T1N2M0, right superior lobe
        # classification_cancer : adenocarcinoma(1), squamous cell carcinoma(2)
        # ----------------------------- ex) adenocarcinoma
        
        df_idx_replace = self.df_replace(df_idx)
        information_used = self.clinical_features
        
        if self.args.CI_input_version == 'v1':
            info_vector = []
            for i in range(len(information_used)):
                if information_used[i] == 'classification_cancer':
                    info_vector.append(df_idx_replace['classification cancer'])
                else:
                    info_vector.append(df_idx_replace[information_used[i]])
            input_CI = torch.from_numpy(np.array(info_vector)[None,:]).float()
            if self.args.model_CI[-1] == 'd': # means "duplicated"
                input_CI_orig = copy.deepcopy(input_CI)
                for c in range(math.floor(512/input_CI.shape[1])):
                    input_CI = np.concatenate((input_CI, input_CI_orig), axis=1)
            
        elif self.args.CI_input_version == 'v2':
            info_vector = []
            for i in range(len(information_used)):
                if information_used[i] == 'age':
                    info_vector.append(df_idx_replace['age'])
                elif information_used[i] == 'classification_cancer':
                    for j in range(len(df_idx_replace['classification cancer'])):
                        info_vector.append(df_idx_replace['classification cancer'][j])
                else:
                    for j in range(len(df_idx_replace[information_used[i]])):
                        info_vector.append(df_idx_replace[information_used[i]][j])
            input_CI = torch.from_numpy(np.array(info_vector)[None,:]).float()
            if self.args.model_CI[-1] == 'd': # means "duplicated"
                input_CI_orig = copy.deepcopy(input_CI)
                for c in range(math.floor(512/input_CI.shape[1])):
                    input_CI = np.concatenate((input_CI, input_CI_orig), axis=1)
            
        elif self.args.CI_input_version == 'text':
        
            if df_idx_replace['sex'] == 0:
                sex = 'male'
            elif df_idx_replace['sex'] == 1:
                sex = 'female'
                
            if df_idx_replace['sm'] == 0:
                smoke = 'nonsmoker'
            elif df_idx_replace['sm'] == 1:
                smoke = 'smoker'

            # locationcancer : right superior lobe(1), right middle lobe(2), right inferior lobe(3), left superior lobe(4), left inferior lobe(5)
            if df_idx_replace['locationcancer'] == 1:
                locationcancer = 'right superior lobe'
            elif df_idx_replace['locationcancer'] == 2:
                locationcancer = 'right middle lobe'
            elif df_idx_replace['locationcancer'] == 3:
                locationcancer = 'right inferior lobe'
            elif df_idx_replace['locationcancer'] == 4:
                locationcancer = 'left superior lobe'
            elif df_idx_replace['locationcancer'] == 5:
                locationcancer = 'left inferior lobe'
            
            if df_idx_replace['classification cancer'] == 1:
                cancerType = 'adenocarcinoma'
            elif df_idx_replace['classification cancer'] == 2:
                cancerType = 'squamous cell carcinoma'
            
            if self.args.CI_prompt_version == 'single':
                clinic = ["%d years old %s %s lung cancer patient, stage %d, T%dN%dM%d, location %s, type %s" %(df_idx['age'], sex, smoke, df_idx['cancerimaging'], df_idx['cancerimagingT'], df_idx['cancerimagingN'], df_idx['cancerimagingM'], locationcancer, cancerType)]
            elif self.args.CI_prompt_version == 'devided':
                clinic = ["a photo of lung cancer patient",
                        "a photo of %d years old" % df_idx['age'],
                        "a photo of %s" % sex,
                        "a photo of %s" % smoke,
                        "a photo of stage %d" % df_idx['cancerimaging'],
                        "a photo of T stage %d" % df_idx['cancerimagingT'],
                        "a photo of N stage %d" % df_idx['cancerimagingN'],
                        "a photo of M stage %d" % df_idx['cancerimagingM'],
                        "a photo of %s" % locationcancer,
                        "a photo of %s" % cancerType]
            
            if self.args.learnablePrompt:
                # clinic = clinic.replace(",", "")
                prompt_prefix = " ".join(["X"] * self.args.n_ctx)
                prompts = [prompt_prefix + " " + info + "." for info in clinic]
            else:
                prompts = [info.replace(",","") + "." for info in clinic]
            
            tokenized_prompts = torch.cat([clip.tokenize(p, context_length=77-self.args.prompt_len) for p in prompts])
            
            input_CI = tokenized_prompts
            
            # if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
            #     clinic = "%d years old %s %s lung cancer patient, stage %d, T%dN%dM%d, location %s, type %s" %(df_idx['age'], sex, smoke, df_idx['cancerimaging'], df_idx['cancerimagingT'], df_idx['cancerimagingN'], df_idx['cancerimagingM'], locationcancer, cancerType)
            #     input_CI = clip.tokenize(clinic, context_length=77-self.args.prompt_len)
            # elif 'CT' in self.args.modality:
            #     # clinic_CT = "%d year old %s %s, lung cancer stage %d, T%dN%dM%d, %s" %(df_idx['age'], sex, smoke, df_idx['cancerimaging'], df_idx['cancerimagingT'], df_idx['cancerimagingN'], df_idx['cancerimagingM'], locationcancer) # AUC 0.8572
            #     clinic_CT = "%d years old %s %s, lung cancer stage %d, T%dN%dM%d, %s" %(df_idx['age'], sex, smoke, df_idx['cancerimaging'], df_idx['cancerimagingT'], df_idx['cancerimagingN'], df_idx['cancerimagingM'], locationcancer)
            #     input_CI = clip.tokenize(clinic_CT, context_length=77-self.args.prompt_len)
            # elif 'pathology' in self.args.modality:
            #     if self.args.pathology_info_version == 'v1':
            #         clinic_pathology = "%d years old %s %s, lung cancer type %s" %(df_idx['age'], sex, smoke, cancerType)
            #     elif self.args.pathology_info_version == 'v2':
            #         clinic_pathology = "%d years old %s %s, lung cancer stage %d, T%dN%dM%d, %s" %(df_idx['age'], sex, smoke, df_idx['cancerimaging'], df_idx['cancerimagingT'], df_idx['cancerimagingN'], df_idx['cancerimagingM'], locationcancer)
            #     elif self.args.pathology_info_version == 'v3':
            #         clinic_pathology = "%d years old %s %s, lung cancer type %s, lung cancer stage %d, T%dN%dM%d, %s" %(df_idx['age'], sex, smoke, cancerType, df_idx['cancerimaging'], df_idx['cancerimagingT'], df_idx['cancerimagingN'], df_idx['cancerimagingM'], locationcancer)
            #     input_CI = clip.tokenize(clinic_pathology, context_length=77-self.args.prompt_len)
        
        
        #%%
        if 'wMask' in self.args.model_CT:
            mask = torch.from_numpy(mask).float()
            if ('CT' in self.args.modality) & ('pathology' in self.args.modality):
                return input_CT, input_pathology, input_CI, label, mask
            elif ('CT' in self.args.modality):
                return input_CT, input_CI, label, mask
            elif ('pathology' in self.args.modality):
                return input_pathology, input_CI, label, mask
        else:
            if ('CT' in self.args.modality) & ('pathology' in self.args.modality):
                return input_CT, input_pathology, input_CI, label
            elif ('CT' in self.args.modality):
                return input_CT, input_CI, label
            elif ('pathology' in self.args.modality):
                if self.args.model_pathology == 'ABMIL_v2':
                    return input_pathology, input_CI, label, BpRc_class
                else:
                    return input_pathology, input_CI, label
            elif ('CI' in self.args.modality):
                return input_CI, label


    def preprocessing_with_metadata(self, img, metadata):
        if ('RescaleSlope' in metadata) and ('RescaleIntercept' in metadata):
            img = (img * metadata.RescaleSlope) + metadata.RescaleIntercept

        img[img>=1000] = 1000
        img[img<=-1024] = -1024

        # img = (img / 2**metadata.BitsStored)

        # if('WindowCenter' in metadata):
        #     if(type(metadata.WindowCenter) == pydicom.multival.MultiValue):
        #         window_center = float(metadata.WindowCenter[0])
        #         window_width = float(metadata.WindowWidth[0])
        #         lwin = window_center - (window_width / 2.0)
        #         rwin = window_center + (window_width / 2.0)
        #     else:
        #         window_center = float(metadata.WindowCenter)
        #         window_width = float(metadata.WindowWidth)
        #         lwin = window_center - (window_width / 2.0)
        #         rwin = window_center + (window_width / 2.0)
        # else:
        #     lwin = np.min(img)
        #     rwin = np.max(img)

        # img[np.where(img < lwin)] = lwin
        # img[np.where(img > rwin)] = rwin
        # img = img - lwin

        if(metadata.PhotometricInterpretation == 'MONOCHROME1'):
            # img[np.where(img < lwin)] = lwin
            # img[np.where(img > rwin)] = rwin
            # img = img - lwin
            # img = 1.0 - img
            img = 2**metadata.BitsStored - img

        # return img / 1024 # in [-1,1)
        return (img + 1024) / (1000 + 1024) # in [0,1)
    
    def data_selection_wLabel(self, df):
        date_standard = 'treatedate'
        # date_standard = 'initialdate'
        survival_type = self.args.survival_type
        year = self.args.year
        
        # 365 --> 365.25 ?? 윤년
        df['label'] = 3
        if survival_type == 'OS':
            duration = abs(pd.to_datetime(df['lastdate']) - pd.to_datetime(df[date_standard])).dt.days
            df.loc[(duration >= year * 365) & (df['dead'] == 0), 'label'] = 0 # negative
            
            df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 1), 'label'] = 1 # positive
            df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 2), 'label'] = 2 # excluded
            df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['hospital'] == 'EUMC'), 'label'] = 1 # positive for EUMC
            
            # if df['hospital'][idx] == 'EUMC':
            #     df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1), 'label'] = 1 # positive
            # else:
            #     df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 1), 'label'] = 1 # positive
            #     df.loc[(duration < year * 365) & (duration > 0) & (df['dead'] == 1) & (df['deathsign'] == 2), 'label'] = 2 # excluded
        elif survival_type == 'RFS':
            duration = abs(pd.to_datetime(df['lastdate']) - pd.to_datetime(df[date_standard])).dt.days
            df.loc[(duration >= year * 365) & (df['relapse'] == 1), 'label'] = 0 # negative
            df.loc[(duration < year * 365) & (duration > 0) & (df['relapse'] != 1), 'label'] = 1 # positive
        
        df_new = df.loc[df['label'].isin([0, 1])]
        
        return df_new
    
    def data_selection(self, df):
        df1 = df[df['classification cancer'].isin([1,2])]
        
        df1 = df1[df1['cancerimaging'].isin([1,2,3,4,'1','2','3','4','1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c'])]
        df1 = df1[df1['cancerimagingT'].isin([1,2,3,4,'1','2','3','4','1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c'])]
        df1 = df1[df1['cancerimagingN'].isin([0,1,2,3,4,'0','1','2','3','4','1a','1b','1c','2a','2b','2c','3a','3b','3c','4a','4b','4c'])]
        df2 = df1[df1['cancerimagingM'].isin([0,1,'0','1','1a','1b','1c'])]
        
        df2 = df2.loc[df2['sex'].isin(['M', 'F'])]
        df2 = df2.loc[df2['sm'].isin(['N', 'Y'])]
        df3 = df2.loc[df2['locationcancer'].isin([1,2,3,4,5])]
        
        df3['label_T'] = 0
        df3.loc[df3['cancerimagingT'].isin([3,'3','3a','3b','3c',4,'4','4a','4b','4c']), 'label_T'] = 1
        
        df3['label_TNM'] = 0
        df3.loc[df3['cancerimaging'].isin([3,'3','3a','3b','3c',4,'4','4a','4b','4c']), 'label_TNM'] = 1

        return df3
    
    def df_replace(self, df):
        df.replace(['M','N','n'], 0, inplace=True)
        df.replace(['F','Y','y'], 1, inplace=True)
        df.replace(['1a','1b','1c'], 1, inplace=True)
        df.replace(['2a','2b','2c'], 2, inplace=True)
        df.replace(['3a','3b','3c'], 3, inplace=True)
        df.replace(['4a','4b','4c'], 4, inplace=True)
        # df.fillna(0, inplace=True)
        df['age'] = 2023 - pd.to_datetime(df['birth date']).year # pd.Timestamp.now().year
        
        # ['sex', 'age', 'sm', 'locationcancer', 'cancerimaging', 'cancerimagingT', 'cancerimagingN', 'cancerimagingM', 'initialdate', 'treatedate', 'relapse', 'lastdate', 'classification_cancer']
        keys = df.keys()
        
        if self.args.CI_input_version == 'v1':
            # if 'sex' in keys:
            if 'age' in keys:
                df['age'] = (df['age'] - 30) / 90 # min 39, max 117
            # if 'sm' in keys:
            if 'locationcancer' in keys:
                df['locationcancer'] = df['locationcancer'] / 5 # 1,2,3,4,5
            if 'cancerimaging' in keys:
                df['cancerimaging'] = df['cancerimaging'] / 4 # 1,2,3,4
            if 'cancerimagingT' in keys:
                df['cancerimagingT'] = df['cancerimagingT'] / 4 # 1,2,3,4
            if 'cancerimagingN' in keys:
                df['cancerimagingN'] = df['cancerimagingN'] / 4 # 0,1,2,3,4
            if 'cancerimagingM' in keys:
                df['cancerimagingM'] = df['cancerimagingM'] # 0,1
            if 'classification_cancer' in keys:
                df['classification cancer'] = df['classification cancer'] / 2 # 1,2
        
        elif self.args.CI_input_version == 'v2':
            if 'sex' in keys:
                df['sex'] = np.eye(2)[df['sex']]
            if 'age' in keys:
                df['age'] = (df['age'] - 30) / 90 # min 39, max 117
            if 'sm' in keys:
                df['sm'] = np.eye(2)[df['sm']]
            if 'locationcancer' in keys:
                df['locationcancer'] = np.eye(5)[df['locationcancer']-1]
            if 'cancerimaging' in keys:
                df['cancerimaging'] = np.eye(4)[df['cancerimaging']-1]
            if 'cancerimagingT' in keys:
                df['cancerimagingT'] = np.eye(4)[df['cancerimagingT']-1]
            if 'cancerimagingN' in keys:
                df['cancerimagingN'] = np.eye(5)[df['cancerimagingN']]
            if 'cancerimagingM' in keys:
                df['cancerimagingM'] = np.eye(2)[df['cancerimagingM']]
            if 'classification cancer' in keys:
                df['classification cancer'] = np.eye(2)[df['classification cancer']-1]
        
        # if 'initialdate' in keys:
        #     df['initialdate'] = (float(df['initialdate'].strftime("%Y.%m%d")) - 2000) / 24 # 2000 ~ 2022년
        # if 'treatedate' in keys:
        #     df['treatedate'] = (float(df['treatedate'].strftime("%Y.%m%d")) - 2000) / 24 # 2000 ~ 2022년
        # if 'relapse' in keys:
        #     df['relapse'] = df['relapse'] / 3  # 1,2,3
        # if 'lastdate' in keys:
        #     df['lastdate'] =  (float(df['lastdate'].strftime("%Y.%m%d")) - 2000) / 24 # 2000 ~ 2023년
        
        return df