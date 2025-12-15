import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Teacher(nn.Module): 
    def __init__(self):
        super(ResNet18Teacher, self).__init__()
        backbone = models.resnet18(pretrained=True)
        
        self.conv1 = backbone.conv1 
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  
        self.layer2 = backbone.layer2 
        self.layer3 = backbone.layer3  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        feat1 = self.layer1(x)   
        feat2 = self.layer2(feat1)  
        feat3 = self.layer3(feat2) 
        
        return feat1, feat2, feat3

class Student(nn.Module): 
    def __init__(self, in_channels=128, out_channels=256):
        super(Student, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ) 
    def forward(self, feat2):
        return self.encoder(feat2)


class Autoencoder(nn.Module): 
    def __init__(self, in_channels=128, out_channels=512):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels//4, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding=1),
        )
    def forward(self, feat2):
        encoded = self.encoder(feat2)  
        decoded = self.decoder(encoded) 
        return decoded

class FusionConv(nn.Module): 
    def __init__(self, st_channels=256, ae_channels=256, out_channels=64):
        super(FusionConv, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(st_channels * 2, st_channels * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(st_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(st_channels, st_channels, kernel_size=3, padding=1),
        )     
    def forward(self, st_out, ae_out, feat1, feat2, feat3):
        fused = torch.cat([st_out, ae_out], dim=1)  
        fused = self.fusion(fused)  
        return fused
        
class AnomalyDetector(nn.Module):
    def __init__(self, decoder_channels=256, teacher_channels=256):
        super(AnomalyDetector, self).__init__()
        self.detector = nn.Sequential(
            nn.Conv2d(decoder_channels + teacher_channels, (decoder_channels + teacher_channels)//4, kernel_size=3, padding=1),
            nn.BatchNorm2d((decoder_channels + teacher_channels)//4,),
            nn.ReLU(inplace=True),
            nn.Conv2d((decoder_channels + teacher_channels)//4, (decoder_channels + teacher_channels)//8, kernel_size=3, padding=1),
            nn.BatchNorm2d((decoder_channels + teacher_channels)//8),
            nn.ReLU(inplace=True),
            nn.Conv2d((decoder_channels + teacher_channels)//8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, decoder_out, teacher_feat1):
        combined = torch.cat([decoder_out, teacher_feat1], dim=1)
        anomaly_map = self.detector(combined)
        return anomaly_map
