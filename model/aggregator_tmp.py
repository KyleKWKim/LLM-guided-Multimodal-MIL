import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from .sam.transformer import TwoWayTransformer

class aggregator(nn.Module):
    def __init__(self, args):
        super(aggregator, self).__init__()
        
        self.args = args
        
        embedding_dim = 512
        
        if 'CT' in self.args.modality:
            weights = 'DEFAULT'
            progress = True
            if self.args.model_CT == 'resnet2plus1d_18':
                from .dim3 import Resnet2plus1D_18
                self.extractor_CT = Resnet2plus1D_18(self.args, weights=weights, progress=progress)
            elif self.args.model_CT == 'resnetMC3_18':
                from .dim3 import ResnetMC3_18
                self.extractor_CT = ResnetMC3_18(self.args, weights=weights, progress=progress)
            elif self.args.model_CT == 'medicalNet':
                from .dim3 import medicalNet
                self.extractor_CT = medicalNet(self.args)
            elif self.args.model_CT == 'SwinUNETR':
                from .dim3 import SwinUNETR
                self.extractor_CT = SwinUNETR(self.args, weights=weights, progress=progress)
            elif self.args.model_CT == 'MViT':
                from .dim3 import MViT_v2
                self.extractor_CT = MViT_v2(self.args, weights=weights, progress=progress)
            
            self.TwoWayTransformer_CT = TwoWayTransformer(
                args=self.args,
                depth=2,
                embedding_dim=embedding_dim,
                num_heads=8,
                mlp_dim=2048,
            )
            
        self.fc_CI2CT = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
                
        if 'pathology' in self.args.modality:
            self.fc_pathology = nn.Sequential(nn.Linear(768, embedding_dim), nn.Tanh())
            if self.args.model_pathology == 'ABMIL':
                from .dim1 import ABMIL
                self.extractor_pathology = ABMIL(args, L=embedding_dim)
            elif self.args.model_pathology == 'ABMIL_v2':
                from .dim1 import ABMIL_v2
                self.extractor_pathology = ABMIL_v2(args, L=embedding_dim)
            elif self.args.model_pathology == 'TransMIL':
                from .dim1 import TransMIL
                self.extractor_pathology = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
            
            self.TwoWayTransformer_Pth = TwoWayTransformer(
                args=self.args,
                depth=2,
                embedding_dim=embedding_dim,
                num_heads=8,
                mlp_dim=2048,
            )
            
        self.fc_CI2Pth = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())

        self.fc_CI = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        
        self.TwoWayTransformer_Both = TwoWayTransformer(
            args=self.args,
            depth=2,
            embedding_dim=embedding_dim,
            num_heads=8,
            mlp_dim=2048,
        )
        
        
        if self.args.aggregator == 'ABMIL':
            from .dim1 import ABMIL
            self.aggregator = ABMIL(args, L=embedding_dim)
        elif self.args.aggregator == 'ABMIL_v2':
            from .dim1 import ABMIL_v2
            self.aggregator = ABMIL_v2(args, L=embedding_dim)
        elif self.args.aggregator == 'TransMIL':
            from .dim1 import TransMIL
            self.aggregator = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
        elif self.args.aggregator == 'TransMIL_seperate':
            from .dim1 import TransMIL
            if 'CT' in self.args.modality:
                self.aggregator_CT = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
            if 'pathology' in self.args.modality:
                self.aggregator_Pth = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
            # self.aggregator = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
            from .dim1 import ABMIL
            self.aggregator = ABMIL(args, L=embedding_dim)
        
        
        # max_seq_len = 15592
        max_seq_len = 100000
        self.pe = torch.zeros((max_seq_len, embedding_dim))
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embedding_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embedding_dim)))
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = self.pe.unsqueeze(0)
        
        if self.args.model_CI == 'simpleFCs_v1':
            from .dim1 import simpleFCs_v1
            self.clinic_extractor = simpleFCs_v1(args)
        elif self.args.model_CI == 'simpleFCs_v1d':
            from .dim1 import simpleFCs_v1d
            self.clinic_extractor = simpleFCs_v1d(args)
        elif self.args.model_CI == 'simpleFCs_v2':
            from .dim1 import simpleFCs_v2
            self.clinic_extractor = simpleFCs_v2(args)
        elif self.args.model_CI == 'simpleFCs_v2d':
            from .dim1 import simpleFCs_v2d
            self.clinic_extractor = simpleFCs_v2d(args)
        elif self.args.model_CI == 'CLIP':
            from .dim1 import CLIP
            if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
                self.clinic_extractor_CT = CLIP(args)
                self.clinic_extractor_Pth = CLIP(args)
            else:
                self.clinic_extractor = CLIP(args)
        # self.prompt = nn.Parameter(torch.randn(2, self.args.prompt_len, embedding_dim))
        self.prompt_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        # self.prompt_embedding = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim)
        
        
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, self.args.num_classes)
        )
            
        
    def forward(self, x_list, x_CI):
        # x_list = [x_CT, x_pathology] CT/pathology 순서대로
        # x_CI : (1, 77-prompt_len)
        
        if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
            # x_CT : (160, 512, 512) --> (1, 512, 160, 14, 14)
            x_input_CT = self.extractor_CT(x_list[0])
            x_input_pathology = self.fc_pathology(x_list[1])
        elif 'CT' in self.args.modality:
            # # x_CT : (160, 512, 512) --> (1, 1, 512)
            # x_input = self.extractor_CT(x_list[0]).unsqueeze(1)
            # x_CT : (160, 512, 512) --> (1, 512, 160, 32, 32)
            x_input = self.extractor_CT(x_list[0])
        elif 'pathology' in self.args.modality:
            # x_pathology : (1, 00, 768) --> (1, 00, 512)
            x_input = self.fc_pathology(x_list[0])
        
        if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
            x_CI_prompted_CT = self.clinic_extractor_CT(x_CI) # (1, 1, 512)
            x_CI_prompted_Pth = self.clinic_extractor_Pth(x_CI) # (1, 1, 512)
        else:
            x_CI_prompted = self.clinic_extractor(x_CI) # (1, 1, 512)
        
        # x_CI_prompted = torch.cat([self.prompt_embedding.unsqueeze(0).expand(x_CI_prompted.shape[0], -1, -1), x_CI_prompted], dim=1)
        
        if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
            b,t,c,h,w = x_input_CT.shape
            if self.args.model_CT == 'resnetMC3_18':
                x_CT2CI, x_CI2CT = self.TwoWayTransformer_CT(x_input_CT, self.pe[:,:c].cuda(), self.fc_CI2CT(x_CI_prompted_CT))
            elif self.args.model_CT == 'medicalNet':
                x_CT2CI, x_CI2CT = self.TwoWayTransformer_CT(x_input_CT, self.pe[:,:c*h*w].cuda(), self.fc_CI2CT(x_CI_prompted_CT))
            
            x_Pth2CI, x_CI2Pth = self.TwoWayTransformer_Pth(x_input_pathology, self.pe[:,:x_input_pathology.shape[1]].cuda(), self.fc_CI2Pth(x_CI_prompted_Pth))
            
            if self.args.aggregator == 'TransMIL_seperate':
                x_CI2CT = self.aggregator_CT(x_CI2CT)[:,None,:]
                x_CI2Pth = self.aggregator_Pth(x_CI2Pth)[:,None,:]
            x0 = torch.cat([x_CT2CI, x_CI2CT, x_Pth2CI, x_CI2Pth], dim=1)
        
        elif ('CT' in self.args.modality):
            b,t,c,h,w = x_input_CT.shape
            if self.args.model_CT == 'resnetMC3_18':
                # x_CT2CI, x_CI2CT = self.TwoWayTransformer_CT(x_input_CT, self.pe[:,:c].cuda(), x_CI_prompted)
                x_CT2CI, x_CI2CT = self.TwoWayTransformer_CT(x_input_CT, self.pe[:,:c].cuda(), self.fc_CI2CT(x_CI_prompted))
            elif self.args.model_CT == 'medicalNet':
                # x_CT2CI, x_CI2CT = self.TwoWayTransformer_CT(x_input_CT, self.pe[:,:c*h*w].cuda(), x_CI_prompted)
                x_CT2CI, x_CI2CT = self.TwoWayTransformer_CT(x_input_CT, self.pe[:,:c*h*w].cuda(), self.fc_CI2CT(x_CI_prompted))
            
            x0 = torch.cat([x_CT2CI, x_CI2CT], dim=1)
            
            # x0 = x_input_CT.mean(dim=(3,4)).permute(0, 2, 1)
            # x0 = x_input_CT.squeeze(dim=(2,3,4))

        elif ('pathology' in self.args.modality):
            x_Pth2CI, x_CI2Pth = self.TwoWayTransformer_Pth(x_input_pathology, self.pe[:,:x_input_pathology.shape[1]].cuda(), self.fc_CI2Pth(x_CI_prompted))
            
            x0 = torch.cat([x_Pth2CI, x_CI2Pth], dim=1)
        
        elif ('CI' in self.args.modality):
            x0 = self.fc_CI(x_CI_prompted)
        
        
        if self.args.aggregator != '-':
            x0 = self.aggregator(x0)
        x = torch.sigmoid(self.fc(x0))
        
        if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
            return x, x_CT2CI, x_Pth2CI, x_CI2CT, x_CI2Pth
        elif ('CT' in self.args.modality):
            return x, x_CT2CI
        elif ('pathology' in self.args.modality):
            return x, x_Pth2CI
        elif ('CI' in self.args.modality):
            return x