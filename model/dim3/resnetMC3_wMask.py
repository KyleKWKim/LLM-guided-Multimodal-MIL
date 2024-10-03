import torch
import torch.nn as nn
import torchvision.models as models

class ResnetMC3_18_wMask(nn.Module):
    def __init__(self, args, weights='DEFAULT', progress=True):
        super(ResnetMC3_18_wMask, self).__init__()

        self.args = args
        self.downsampling = nn.Conv3d(in_channels=2, out_channels=3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.model = models.video.mc3_18(weights=weights, progress=progress)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_features, num_features))

    def forward(self, x, mask):
        self.model(self.downsampling(torch.cat([x,mask],dim=1)))