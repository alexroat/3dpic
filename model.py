import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch3d.transforms


print("MPS available:",torch.backends.mps.is_available())
print("MPS built:",torch.backends.mps.is_built())
device = torch.device("mps")
print(device)
dtype = torch.float

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.linear = nn.Linear(256*128*128, 6)

    def forward(self, x):
        # #print(x.shape)
        #x=x.permute(2,1,0).unsqueeze(0)
        #print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.pool3(F.relu(self.conv3(x)))
        #print(x.shape)
        s = F.relu(self.conv4(x))
        # #print(s.shape)
        s=s.view(3, 128*128)
        #print(x.shape)
        p = self.linear(torch.flatten(x, 1))
        return s, p

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, s, p):
        s = s.to(device)
        p = p.to(device)
        # calcola matrice di rotazione
        rotation_matrices = pytorch3d.transforms.euler_angles_to_matrix(p[:, :3], "XYZ").to(device)
        x = torch.matmul(rotation_matrices, s)
        x = x + p[:, 3:].t()
        x = s.view(1, 3, 128, 128)
        x = F.relu(self.conv1(x))
        x = self.upsample2(F.relu(self.conv2(x)))
        x = self.upsample3(F.relu(self.conv3(x)))
        x = self.upsample4(F.relu(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        #x = x.squeeze().permute(2,1,0)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        s, p = self.encoder(x)
        #print(s.shape,p.shape)
        x = self.decoder(s, p)
        return x
