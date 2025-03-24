import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure


# ---- STEP 1: CUSTOM DATASET ----
class CustomDataset(Dataset):
    def __init__(self, hqFolder, lqFolder, lqSize=(100, 100), hqSize=(200, 200)):  
        self.hqImages = sorted(glob.glob(os.path.join(hqFolder, "**", "*.png"), recursive=True) +
                               glob.glob(os.path.join(hqFolder, "**", "*.jpg"), recursive=True))
        self.lqImages = sorted(glob.glob(os.path.join(lqFolder, "**", "*.png"), recursive=True) +
                               glob.glob(os.path.join(lqFolder, "**", "*.jpg"), recursive=True))
        
        assert len(self.hqImages) == len(self.lqImages), "Mismatch between HQ and LQ image counts"

        self.hqTransform = transforms.Compose([
            transforms.Resize(hqSize),  
            transforms.ToTensor()
        ])
        self.lqTransform = transforms.Compose([
            transforms.Resize(lqSize),  
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hqImages)

    def __getitem__(self, index):
        hqImg = Image.open(self.hqImages[index]).convert('RGB')
        lqImg = Image.open(self.lqImages[index]).convert('RGB')
        return self.lqTransform(lqImg), self.hqTransform(hqImg)

# Create DataLoader
highquality_folder = r"C:\Users\ashut\Documents\GitHub\2dshapedetection-major\ElementaryCQT_200x200"
lowquality_folder = r"C:\Users\ashut\Documents\GitHub\2dshapedetection-major\ElementaryCQT_100x100"

trainDataset = CustomDataset(highquality_folder, lowquality_folder)  
trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True)

# ---- STEP 2: EDSR MODEL ----
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual  # Residual connection

class EDSR(nn.Module):
    def __init__(self, numResBlocks=16, upscaleFactor=2):
        super(EDSR, self).__init__()
        self.inputConv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.resBlocks = nn.Sequential(*[ResidualBlock(64) for _ in range(numResBlocks)])
        self.outputConv = nn.Conv2d(64, 3 * (upscaleFactor ** 2), kernel_size=3, padding=1)
        self.pixelShuffle = nn.PixelShuffle(upscaleFactor)

    def forward(self, x):
        x = self.inputConv(x)
        x = self.resBlocks(x)
        x = self.outputConv(x)
        x = self.pixelShuffle(x)  # Upscales from 100x100 → 200x200
        return x

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EDSR().to(device)


# ---- STEP 3: DEFINE LOSS & OPTIMIZER ----
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        return torch.mean(torch.sqrt((preds - targets) ** 2 + self.epsilon ** 2))

# Instantiate loss functions & optimizer
charbLoss = CharbonnierLoss()
ssimLoss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- STEP 4: TRAINING LOOP ----
numEpochs = 4  # Adjust as needed
for epoch in range(numEpochs):
    model.train()
    epochLoss = 0
    for lrImgs, hrImgs in tqdm(trainLoader):
        lrImgs, hrImgs = lrImgs.to(device), hrImgs.to(device)
        optimizer.zero_grad()
        preds = model(lrImgs)
        preds = F.interpolate(preds, size=(200, 200), mode='bilinear', align_corners=False)  # Ensure predictions match HR size
        loss1 = charbLoss(preds, hrImgs)  # Compute Charbonnier loss
        loss2 = 1 - ssimLoss(preds, hrImgs)  # Compute SSIM loss
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        epochLoss += loss.item()
    print(f"Epoch {epoch+1}/{numEpochs} - Loss: {epochLoss:.4f}")

# Save trained model
torch.save(model.state_dict(), "EDSR.pth")
print("Model saved!")

# # Prevent CMD from closing immediately
# input("Press Enter to exit...")
