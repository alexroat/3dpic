import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

print("MPS available:",torch.backends.mps.is_available())
print("MPS built:",torch.backends.mps.is_built())
device = torch.device("mps")
print(device)
dtype = torch.float

CS=60

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 50 * 30, CS)

    def forward(self, x):
        print(x.shape)
        #x = x.permute(0, 3, 1, 2)  # Permuta le dimensioni per adattarle alla convenzione PyTorch (batch, canali, altezza, larghezza)
        x=x.permute(2,1,0).unsqueeze(0)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(CS, 256 * 50 * 30)
        self.upsc4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsc3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsc2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsc1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 50, 30)
        x = self.upsc4(x)
        x = torch.relu(self.conv4(x))
        x = self.upsc3(x)
        x = torch.relu(self.conv3(x))
        x = self.upsc2(x)
        x = torch.relu(self.conv2(x))
        x = self.upsc1(x)
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(x)  # Utilizzare la sigmoide per garantire che i valori siano nell'intervallo [0, 1]
        x = x.squeeze(0).permute(2, 1, 0)
        print(x.shape)
        return x



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
