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
# CONFIGURATION
# ==========================================
# ASVspoof 2019 logical access dataset paths (Update these!)
REAL_AUDIO_DIR = r"Dataset\ASVspoof_LA\Real"
FAKE_AUDIO_DIR = r"Dataset\ASVspoof_LA\Fake"
MODEL_SAVE_PATH = "models/tinylstm_audio_v1.pth"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audio parameters
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 3.0  # Seconds (Truncate/Pad to this length to standardize LSTM sequence)
FAKE_PENALTY_WEIGHT = 4.0 # (Step D)

# ==========================================
# STEP A & B: VAD AND LFCC EXTRACTION
# ==========================================
class AudioForensicDataset(Dataset):
    def __init__(self, real_dir, fake_dir, max_duration=3.0, sample_rate=16000):
        self.file_paths = []
        self.labels = []
        self.max_length = int(sample_rate * max_duration)
        self.sample_rate = sample_rate

        # Load file paths
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.wav') or f.endswith('.flac'):
                    self.file_paths.append(os.path.join(real_dir, f))
                    self.labels.append(1.0) # Real = 1
                    
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.wav') or f.endswith('.flac'):
                    self.file_paths.append(os.path.join(fake_dir, f))
                    self.labels.append(0.0) # Fake = 0

        # STEP B: The LFCC Extractor (Linear Frequency Cepstral Coefficients)
        # Why LFCC? It doesn't compress high frequencies like MFCC does. 
        # AI generators struggle with high-frequency phase alignment.
        self.lfcc_transform = T.LFCC(
            sample_rate=self.sample_rate,
            n_lfcc=40, # Extract 40 frequency bands
            speckwargs={"n_fft": 512, "hop_length": 256, "center": False}
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio using librosa (easier to trim silence)
        waveform, sr = librosa.load(path, sr=self.sample_rate)

        # STEP A: VAD (Voice Activity Detection) - Trimming dead silence
        # This prevents the LSTM from wasting compute on room noise
        waveform, _ = librosa.effects.trim(waveform, top_db=30)

        # Convert back to tensor
        waveform = torch.tensor(waveform).unsqueeze(0) # Shape: [1, Time]

        # Standardize Length (Pad if too short, Truncate if too long)
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Apply LFCC Extraction
        lfcc = self.lfcc_transform(waveform) # Shape: [1, n_lfcc, Time/hop_length]
        
        # LSTM expects input shape: [Sequence_Length, Batch, Features] or [Batch, Seq, Features]
        # We drop the channel dimension and transpose to [Time, Features]
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
    
    # 1. Load Data
    dataset = AudioForensicDataset(REAL_AUDIO_DIR, FAKE_AUDIO_DIR, max_duration=MAX_AUDIO_LENGTH)
    
    if len(dataset) == 0:
        print("[ERROR] No audio files found. Check your dataset paths.")
        return

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Init Model
    model = TinyLSTM(input_size=40, hidden_size=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # STEP D: THE ANTI-LAZY PENALTY
    # Since Fake = 0 and Real = 1, we must dynamically weight the loss for Fake samples.
    # We punish the model heavily if it predicts "Real" when the audio is actually a Voice Clone.
    
    best_val_loss = float('inf')
    
    print("[INFO] Beginning LSTM Training...")
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