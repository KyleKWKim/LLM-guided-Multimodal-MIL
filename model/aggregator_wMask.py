import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class aggregator_wMask(nn.Module):
    def __init__(self, args):
        super(aggregator_wMask, self).__init__()
        
        self.args = args
        
        if 'CT' in self.args.modality:
            weights = 'DEFAULT'
            progress = True
            if self.args.model_CT == 'resnetMC3_18_wMask':
                from .dim3 import ResnetMC3_18_wMask
                self.extractor_CT = ResnetMC3_18_wMask(self.args, weights=weights, progress=progress)
            elif self.args.model_CT == 'SwinUNETR_wMask':
                from .dim3 import SwinUNETR_wMask
                self.extractor_CT = SwinUNETR_wMask(self.args, weights=weights, progress=progress)
        
        if 'pathology' in self.args.modality:
            if self.args.model_pathology == 'ABMIL':
                from .dim1 import gatedAttention
                self.extractor_pathology = gatedAttention(args)
            elif self.args.model_pathology == 'TransMIL':
                from .dim1 import TransMIL
                self.extractor_pathology = TransMIL(n_classes=self.args.num_classes)
        
        if 'CI' in self.args.modality:
            if self.args.model_CI == 'simpleFCs_v1':
                from .dim1 import simpleFCs_v1
                self.extractor_CI = simpleFCs_v1(args)
            elif self.args.model_CI == 'simpleFCs_v2':
                from .dim1 import simpleFCs_v2
                self.extractor_CI = simpleFCs_v2(args)
        
        self.concat_feature_in = 0
        self.concat_feature_out = 0
        if 'CT' in self.args.modality:
            self.concat_feature_in += 768
            # self.concat_feature_in += 512
            self.concat_feature_out += 192
        if 'pathology' in self.args.modality:
            self.concat_feature_in += 768
            self.concat_feature_out += 192
        if 'CI' in self.args.modality:
            self.concat_feature_in += len(self.args.clinical_features)
            # self.concat_feature_out += len(self.args.clinical_features)
        
        if ('CT' not in self.args.modality) & ('pathology' not in self.args.modality) & ('CI' in self.args.modality):
            self.fc = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(self.concat_feature_in, self.args.num_classes)
            )
        # elif ('CT' not in self.args.modality) & ('pathology' in self.args.modality) & ('CI' not in self.args.modality):
        #     self.fc = nn.Sequential(
        #         nn.Dropout(0.25),
        #         nn.Linear(self.concat_feature_in, self.args.num_classes)
        #     )
        # elif ('CT' not in self.args.modality) & ('pathology' in self.args.modality) & ('CI' in self.args.modality):
        #     self.fc = nn.Sequential(
        #         nn.Dropout(0.25),
        #         nn.Linear(self.concat_feature_in, self.args.num_classes)
        #     )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(0.25), nn.Linear(self.concat_feature_in, self.concat_feature_out), nn.ReLU(),
                nn.Dropout(0.25), nn.Linear(self.concat_feature_out, self.args.num_classes)
            )
        
    def forward(self, x_list, mask):
        # x_list = [x_CT, x_pathology, x_CI] CT/pathology/CI 순서대로
        
        if ('CT' in self.args.modality) & ('pathology' in self.args.modality) & ('CI' in self.args.modality):
            if self.args.model_CT == 'SwinUNETR_wMask':
                x_CT    = self.extractor_CT(x_list[0], mask).squeeze(1)
            else:
                x_CT    = self.extractor_CT(torch.cat([x_list[0], mask], dim=1))
            x_pathology = self.extractor_pathology(x_list[1]).squeeze(1)
            x_CI        = self.extractor_CI(x_list[2]).squeeze(1)
            x = torch.cat([x_CT, x_pathology, x_CI], dim=1)
        elif ('CT' in self.args.modality) & ('pathology' in self.args.modality):
            if self.args.model_CT == 'SwinUNETR_wMask':
                x_CT    = self.extractor_CT(x_list[0], mask).squeeze(1)
            else:
                x_CT    = self.extractor_CT(torch.cat([x_list[0], mask], dim=1))
            x_pathology = self.extractor_pathology(x_list[1]).squeeze(1)
            x = torch.cat([x_CT, x_pathology], dim=1)
        elif ('pathology' in self.args.modality) & ('CI' in self.args.modality):
            x_pathology = self.extractor_pathology(x_list[0]).squeeze(1)
            x_CI        = self.extractor_CI(x_list[1]).squeeze(1)
            x = torch.cat([x_pathology, x_CI], dim=1)
        elif ('CT' in self.args.modality) & ('CI' in self.args.modality):
            if self.args.model_CT == 'SwinUNETR_wMask':
                x_CT    = self.extractor_CT(x_list[0], mask).squeeze(1)
            else:
                x_CT    = self.extractor_CT(torch.cat([x_list[0], mask], dim=1))
            x_CI        = self.extractor_CI(x_list[1]).squeeze(1)
            x = torch.cat([x_CT, x_CI], dim=1)
        elif ('CT' in self.args.modality):
            if self.args.model_CT == 'SwinUNETR_wMask':
                x_CT    = self.extractor_CT(x_list[0], mask).squeeze(1)
            else:
                x_CT    = self.extractor_CT(torch.cat([x_list[0], mask], dim=1))
            x = x_CT
        elif ('pathology' in self.args.modality):
            x_pathology = self.extractor_pathology(x_list[0]).squeeze(1)
            x = x_pathology
        elif ('CI' in self.args.modality):
            x_CI        = self.extractor_CI(x_list[0]).squeeze(1)
            x = x_CI
        
        return torch.sigmoid(self.fc(x))