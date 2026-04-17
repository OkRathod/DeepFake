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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_AUDIO_DIR = r"ASVspoof2019_root\LA\ASVspoof2019_LA_train\flac" 
TRAIN_PROTOCOL_FILE = r"ASVspoof2019_root\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

# We will split this DEV set into Validation and Test
DEV_AUDIO_DIR = r"ASVspoof2019_root\LA\ASVspoof2019_LA_dev\flac"
DEV_PROTOCOL_FILE = r"ASVspoof2019_root\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"

MODEL_SAVE_PATH = "models/tinylstm_audio_v1.pth"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 3.0  # Seconds
FAKE_PENALTY_WEIGHT = 4.0 

# ==========================================
# 2. DATASET PARSER
# ==========================================
class AudioForensicDataset(Dataset):
    def __init__(self, audio_dir, protocol_file, max_duration=3.0, sample_rate=16000):
        self.file_paths = []
        self.labels = []
        self.max_length = int(sample_rate * max_duration)
        self.sample_rate = sample_rate
        self.audio_dir = audio_dir

        print(f"[INFO] Parsing Protocol File: {protocol_file}")
        
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                filename = parts[1] + ".flac"
                file_path = os.path.join(self.audio_dir, filename)
                
                label_str = parts[4]
                label = 1.0 if label_str == "bonafide" else 0.0 # Real = 1, Fake = 0
                
                if os.path.exists(file_path):
                    self.file_paths.append(file_path)
                    self.labels.append(label)

        print(f"[INFO] Successfully loaded {len(self.file_paths)} audio paths.")

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

        waveform, sr = librosa.load(path, sr=self.sample_rate)
        waveform, _ = librosa.effects.trim(waveform, top_db=30)
        waveform = torch.tensor(waveform).unsqueeze(0) 

        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        lfcc = self.lfcc_transform(waveform) 
        lfcc = lfcc.squeeze(0).transpose(0, 1) 

        return lfcc, torch.tensor(label, dtype=torch.float32)
    
# ==========================================
# 3. THE TINY-LSTM ARCHITECTURE
# ==========================================
class TinyLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=64, num_layers=2):
        super(TinyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ==========================================
# 4. PLOTTING & EVALUATION FUNCTIONS
# ==========================================
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'g-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tinylstm_training_history.png')
    print("[INFO] Saved training history graph as 'tinylstm_training_history.png'")
    plt.show()

def evaluate_test_set(model, test_loader):
    print("\n[INFO] Starting Final Test Set Evaluation...")
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Testing"):
            sequences = sequences.to(DEVICE)
            labels = labels.view(-1).numpy()
            
            outputs = model(sequences)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
            
            y_true.extend(labels)
            y_pred.extend(preds)
            
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake (0)', 'Real (1)'], 
                yticklabels=['Fake (0)', 'Real (1)'])
    plt.title('TinyLSTM Audio Sentinel - Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('tinylstm_confusion_matrix.png')
    print("[INFO] Saved Confusion Matrix as 'tinylstm_confusion_matrix.png'")
    plt.show()
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

# ==========================================
# 5. MAIN TRAINING ENGINE
# ==========================================
def train():
    print(f"[INFO] Using Device: {DEVICE}")
    
    print("\n--- Loading Training Set ---")
    train_dataset = AudioForensicDataset(TRAIN_AUDIO_DIR, TRAIN_PROTOCOL_FILE, max_duration=MAX_AUDIO_LENGTH)
    
    print("\n--- Loading Development Set (Will be split into Val and Test) ---")
    full_dev_dataset = AudioForensicDataset(DEV_AUDIO_DIR, DEV_PROTOCOL_FILE, max_duration=MAX_AUDIO_LENGTH)
    
    if len(train_dataset) == 0 or len(full_dev_dataset) == 0:
        print("[ERROR] Audio files not found. Please double-check your folder paths.")
        return

    # ✨ NEW: Split the Dev dataset 50/50 into Validation and Test
    val_size = len(full_dev_dataset) // 2
    test_size = len(full_dev_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(full_dev_dataset, [val_size, test_size])
    
    print(f"[INFO] Data Split Complete:")
    print(f"       Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TinyLSTM(input_size=40, hidden_size=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    
    # ✨ NEW: History Trackers for plotting
    history_train_loss, history_val_loss = [], []
    history_train_acc, history_val_acc = [], []
    
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
        avg_train_loss = running_loss / len(train_loader)
        
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
        
        # Save metrics for plotting
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)
        
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("🌟 Best Audio Sentinel Model Saved!")

    # ==========================================
    # 6. POST-TRAINING ANALYSIS
    # ==========================================
    print("\n[INFO] Training Phase Complete. Generating Graphs...")
    plot_training_history(history_train_loss, history_val_loss, history_train_acc, history_val_acc)
    
    # Load the best model to run on the Test Set
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    evaluate_test_set(model, test_loader)

if __name__ == '__main__':
    train()
