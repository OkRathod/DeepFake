import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import io

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
# Ensure these paths are correct for your machine
BASE_DIR = r"/home/score/Downloads/archive(1)/Deepfake_Image_Dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VAL_DIR =   os.path.join(BASE_DIR, "Validation")
TEST_DIR =  os.path.join(BASE_DIR, "Test")

MODEL_SAVE_PATH = "models"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'effnet_b2_best_v1.pth')
LATEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'effnet_b2_latest_v1.pth')

# Optimized for RTX A4000 (16GB) to avoid OOM
BATCH_SIZE = 32 
LEARNING_RATE = 1e-4
EPOCHS = 20
IMG_SIZE = (224, 224) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Augmentation Pipeline
# ==========================================
class JPEGCompression(object):
    def __init__(self, quality_min=40, quality_max=80):
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img):
        if not isinstance(img, Image.Image): return img
        quality = np.random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([JPEGCompression()], p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==========================================
# 3. Model Definition (EfficientNet-B2)
# ==========================================
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # B2 provides a great balance of accuracy and VRAM efficiency
        self.network = models.efficientnet_b2(weights='IMAGENET1K_V1')
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1) 
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. Data Loading Engine
# ==========================================
def get_dataloaders():
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(VAL_DIR, transform=get_transforms(train=False))
    
    test_path = TEST_DIR if os.path.exists(TEST_DIR) else VAL_DIR
    test_ds = datasets.ImageFolder(test_path, transform=get_transforms(train=False))

    # High workers for your i9-12900K
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader

# ==========================================
# 5. Training Engine
# ==========================================
def train_model():
    # Performance & Memory Tuning
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True 
    
    if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)

    model = DeepfakeDetector().to(device)
    
    # PyTorch 2.0+ Compile
    try:
        model = torch.compile(model)
        print("🚀 GPU Optimization Compiled.")
    except:
        pass

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() 

    train_loader, val_loader, _ = get_dataloaders()
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.float().to(device).view(-1, 1)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="[Val]"):
                imgs, labels = imgs.to(device), labels.float().to(device).view(-1, 1)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # --- Metrics ---
        t_loss = train_loss / len(train_loader)
        t_acc = train_correct / train_total
        v_loss = val_loss / len(val_loader)
        v_acc = val_correct / val_total
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f"Epoch {epoch+1}: Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Val Loss: {v_loss:.4f}")

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("🌟 Best Model Saved!")

        torch.save(model.state_dict(), LATEST_MODEL_PATH)

    return history

if __name__ == '__main__':
    train_model()
