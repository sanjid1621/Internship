import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import h5py
from sklearn.model_selection import train_test_split

# 1. Define CNN Model (Improved)
class PQDCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(PQDCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 128),  # fc_layers.0
            nn.ReLU(),             # fc_layers.1
            nn.Linear(128, num_classes),  # fc_layers.2
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# 2. Custom Dataset
class PQDDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx]).unsqueeze(0)  # Add channel dim
        y = torch.LongTensor([self.labels[idx]])
        return x, y

# 3. Load MATLAB Data (Improved)
def load_matlab_data(mat_file):
    try:
        # Try regular loadmat first
        data = loadmat(mat_file)
        features = data['features']
        labels = data['labels']
    except NotImplementedError:
        # Fall back to h5py for v7.3 files
        with h5py.File(mat_file, 'r') as f:
            features = np.array(f['features']).transpose()
            labels = np.array(f['labels']).transpose()
    
    # Reshape and normalize features
    features = features.reshape(-1, features.shape[-1])
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)  # Normalize
    
    # Create labels (assuming 6 classes)
    frame_labels = np.repeat(np.arange(6), features.shape[0]//6)
    
    return features, frame_labels

# 4. Train Function (Improved)
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze())
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.squeeze()).sum().item()
        
        # Print statistics
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Acc: {100*val_correct/val_total:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'pqd_cnn_best.pth')
            print('Best model saved!')

# 5. Main Execution (Improved)
if __name__ == "__main__":
    # Load and split data
    features, labels = load_matlab_data('pqd_frames.mat')
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and dataloaders
    train_dataset = PQDDataset(X_train, y_train)
    val_dataset = PQDDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = PQDCNN(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)
    
    # Save final model
    torch.save(model.state_dict(), 'pqd_cnn_final.pth')
    print("Final model saved as pqd_cnn_final.pth")
    
    # Export to ONNX (only if ONNX is installed)
    try:
        import onnx
        sample_input = torch.randn(1, 1, 256)
        torch.onnx.export(model, sample_input, "pqd_cnn.onnx", 
                         input_names=["input"], output_names=["output"],
                         dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print("Model exported to ONNX format as pqd_cnn.onnx")
    except ImportError:
        print("ONNX not installed. Skipping ONNX export.")