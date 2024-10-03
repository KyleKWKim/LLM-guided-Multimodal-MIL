import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class aggregator(nn.Module):
    def __init__(self, args):
        super(aggregator, self).__init__()
        
        self.args = args
        
        self.concat_feature_in = 0
        self.concat_feature_out = 0
        if 'CT' in self.args.modality:
            self.concat_feature_in_CT = 512
            self.concat_feature_mid_CT = 512
            self.concat_feature_out += self.concat_feature_mid_CT
            
            weights = 'DEFAULT'
            progress = True
            if self.args.model_CT == 'resnet2plus1d_18':
                from .dim3 import Resnet2plus1D_18
                self.extractor_CT = Resnet2plus1D_18(self.args, weights=weights, progress=progress)
            elif self.args.model_CT == 'resnetMC3_18':
                from .dim3 import ResnetMC3_18
                self.extractor_CT = ResnetMC3_18(self.args, weights=weights, progress=progress)
            elif self.args.model_CT == 'SwinUNETR':
                from .dim3 import SwinUNETR
                self.extractor_CT = SwinUNETR(self.args, weights=weights, progress=progress)
            elif self.args.model_CT == 'MViT':
                from .dim3 import MViT_v2
                self.extractor_CT = MViT_v2(self.args, weights=weights, progress=progress)
                
        if 'pathology' in self.args.modality:
            if self.args.model_pathology == 'ABMIL':
                self.concat_feature_in_pathology = 768
                self.concat_feature_mid_pathology = 512
                self.concat_feature_out += self.concat_feature_mid_pathology
                from .dim1 import ABMIL
                self.extractor_pathology = ABMIL(args)
            elif self.args.model_pathology == 'ABMIL_v2':
                self.concat_feature_in_pathology = 768 + 1
                self.concat_feature_mid_pathology = 512
                self.concat_feature_out += self.concat_feature_mid_pathology
                from .dim1 import ABMIL_v2
                self.extractor_pathology = ABMIL_v2(args)
            elif self.args.model_pathology == 'TransMIL':
                self.concat_feature_in_pathology = 512
                self.concat_feature_mid_pathology = 512
                self.concat_feature_out += self.concat_feature_mid_pathology
                from .dim1 import TransMIL
                self.extractor_pathology = TransMIL(n_classes=self.args.num_classes)
        
        
        
        if len(self.args.modality) == 1:
            if 'CT' in self.args.modality:
                self.concat_feature_mid = self.concat_feature_in_CT
            elif 'pathology' in self.args.modality:
                self.concat_feature_mid = self.concat_feature_in_pathology
        elif len(self.args.modality) == 2:
            if 'CT' in self.args.modality:
                self.fc_CT = nn.Sequential(nn.Dropout(0.25),
                                        nn.Linear(self.concat_feature_in_CT, self.concat_feature_mid_CT),
                                        nn.ReLU())
            if 'pathology' in self.args.modality:
                self.fc_pathology = nn.Sequential(nn.Dropout(0.25),
                                        nn.Linear(self.concat_feature_in_pathology, self.concat_feature_mid_pathology),
                                        nn.ReLU())
            self.concat_feature_mid = self.concat_feature_mid_CT
        
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.concat_feature_mid, self.args.num_classes)
        )
            
            
        
    def forward(self, x_list):
        # x_list = [x_CT, x_pathology] CT/pathology 순서대로
        
        if ('CT' in self.args.modality) & ('pathology' in self.args.modality):
            if self.args.model_CT == 'SwinUNETR':
                x_CT    = self.extractor_CT(x_list[0]).squeeze(1)
            elif self.args.model_CT == 'MViT':
                x_CT    = self.extractor_CT(x_list[0].squeeze(1))
            else:
                x_CT    = self.extractor_CT(x_list[0])
            x_CT        = self.fc_CT(x_CT)
            
            x_pathology = self.extractor_pathology(x_list[1]).squeeze(1)
            x_pathology = self.fc_pathology(x_pathology)
            
            x = (x_CT + x_pathology) / 2
            
            return x_CT, x_pathology, torch.sigmoid(self.fc(x))
            
        elif ('CT' in self.args.modality):
            if self.args.model_CT == 'SwinUNETR':
                x_CT    = self.extractor_CT(x_list[0]).squeeze(1)
            elif self.args.model_CT == 'MViT':
                x_CT    = self.extractor_CT(x_list[0].squeeze(1))
            else:
                x_CT    = self.extractor_CT(x_list[0])
            x = x_CT
            
            return x_CT, torch.sigmoid(self.fc(x))
            
        elif ('pathology' in self.args.modality):
            if self.args.model_pathology == 'ABMIL_v2':
                # [pathology, Biopsy/Resection]
                x_pathology = self.extractor_pathology(x_list[0], x_list[1]).squeeze(1)
            else:
                # [pathology]
                x_pathology = self.extractor_pathology(x_list[0]).squeeze(1)
            x = x_pathology
        
            return x_pathology, torch.sigmoid(self.fc(x))