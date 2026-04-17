import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. UPDATED CONFIGURATION
# ==========================================
# Point these directly to the extracted ASVspoof folders
# Make sure to include the '/flac' subfolder if it exists!
TRAIN_AUDIO_DIR = r"ASVspoof2019_root\LA\ASVspoof2019_LA_train\flac" 
TRAIN_PROTOCOL_FILE = r"ASVspoof2019_root\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

DEV_AUDIO_DIR = r"ASVspoof2019_root\LA\ASVspoof2019_LA_dev\flac"
DEV_PROTOCOL_FILE = r"ASVspoof2019_root\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

MODEL_SAVE_PATH = "models/tinylstm_audio_v1.pth"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 3.0  # Seconds
FAKE_PENALTY_WEIGHT = 4.0 # Punish the model 4x harder for missing a deepfake

# ==========================================
# 2. UPDATED DATASET PARSER (Protocol Based)
# ==========================================
class AudioForensicDataset(Dataset):
    def __init__(self, audio_dir, protocol_file, max_duration=3.0, sample_rate=16000):
        self.file_paths = []
        self.labels = []
        self.max_length = int(sample_rate * max_duration)
        self.sample_rate = sample_rate
        self.audio_dir = audio_dir

        print(f"[INFO] Parsing Protocol File: {protocol_file}")
        
        # 1. Read the text file mapping
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Column 2 is the filename (e.g., LA_T_1138215)
                filename = parts[1] + ".flac"
                file_path = os.path.join(self.audio_dir, filename)
                
                # Column 5 is the label (bonafide or spoof)
                label_str = parts[4]
                label = 1.0 if label_str == "bonafide" else 0.0 # Real = 1, Fake = 0
                
                # Only add if the audio file actually exists on the hard drive
                if os.path.exists(file_path):
                    self.file_paths.append(file_path)
                    self.labels.append(label)

        print(f"[INFO] Successfully loaded {len(self.file_paths)} audio paths.")

        # 2. The LFCC Extractor (Linear Frequency Cepstral Coefficients)
        self.lfcc_transform = T.LFCC(
            sample_rate=self.sample_rate,
            n_lfcc=40, 
            speckwargs={"n_fft": 512, "hop_length": 256, "center": False}
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio (Librosa automatically resamples to 16k if needed)
        waveform, sr = librosa.load(path, sr=self.sample_rate)

        # VAD (Voice Activity Detection) - Trimming dead silence
        waveform, _ = librosa.effects.trim(waveform, top_db=30)

        # Convert back to PyTorch tensor
        waveform = torch.tensor(waveform).unsqueeze(0) 

        # Standardize Length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Extract LFCC and reshape for LSTM
        lfcc = self.lfcc_transform(waveform) 
        lfcc = lfcc.squeeze(0).transpose(0, 1) 

        return lfcc, torch.tensor(label, dtype=torch.float32)
    
# ==========================================
# STEP C: THE TINY-LSTM ARCHITECTURE
# ==========================================
class TinyLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=64, num_layers=2):
        super(TinyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The core LSTM layer
        # batch_first=True makes input shape [Batch, Sequence/Time, Features]
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) # Single output logit for Real(1) vs Fake(0)
        )

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # TEMPORAL POOLING: We only care about the final hidden state of the sequence
        # to make our final decision on whether the audio clip is real or fake.
        out = out[:, -1, :] 
        
        # Pass through classifier
        out = self.fc(out)
        return out

# ==========================================
# TRAINING ENGINE
# ==========================================
def train():
    print(f"[INFO] Using Device: {DEVICE}")
    print("[INFO] Preparing Audio Dataset (VAD + LFCC Extraction)...")
    
    # 1. Load Train and Validation Data Separately using the Protocol Files
    print("\n--- Loading Training Set ---")
    train_dataset = AudioForensicDataset(TRAIN_AUDIO_DIR, TRAIN_PROTOCOL_FILE, max_duration=MAX_AUDIO_LENGTH)
    
    print("\n--- Loading Validation Set ---")
    val_dataset = AudioForensicDataset(DEV_AUDIO_DIR, DEV_PROTOCOL_FILE, max_duration=MAX_AUDIO_LENGTH)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("[ERROR] Audio files not found. Please double-check your folder paths.")
        return

    # 2. Create DataLoaders
    # We don't need random_split anymore because ASVspoof provides a dedicated Dev set
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Init Model
    model = TinyLSTM(input_size=40, hidden_size=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # STEP D: THE ANTI-LAZY PENALTY
    best_val_loss = float('inf')
    
    print("\n[INFO] Beginning LSTM Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for sequences, labels in loop:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
            # Dynamic Sample Weighting (Fake = 0 gets the FAKE_PENALTY_WEIGHT)
            sample_weights = torch.where(labels == 0.0, torch.tensor(FAKE_PENALTY_WEIGHT, device=DEVICE), torch.tensor(1.0, device=DEVICE))
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels, weight=sample_weights)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE).view(-1, 1)
                outputs = model(sequences)
                
                sample_weights = torch.where(labels == 0.0, torch.tensor(FAKE_PENALTY_WEIGHT, device=DEVICE), torch.tensor(1.0, device=DEVICE))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels, weight=sample_weights)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                
        val_acc = v_correct / v_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("🌟 Best Audio Sentinel Model Saved!")
            
if __name__ == '__main__':
    train()
