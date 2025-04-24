import numpy as np
import torch
from pqd_cnn import PQDCNN
import os

# Load trained model
model = PQDCNN(num_classes=6)
model.load_state_dict(torch.load('pqd_cnn.pth', weights_only=True))  # Added security flag
model.eval()

# Class labels
class_labels = ["Nominal", "Sag", "Swell", "Harmonics", "Transient", "Fluctuation"]

# Load current frame (robust version)
try:
    # First try numpy's loadtxt
    current_frame = np.loadtxt('current_frame.csv', delimiter=',')
    
    # If empty, try pandas
    if current_frame.size == 0:
        import pandas as pd
        current_frame = pd.read_csv('current_frame.csv', header=None).values
        
    # Convert to tensor
    current_frame = torch.FloatTensor(current_frame).unsqueeze(0).unsqueeze(0)
except Exception as e:
    raise ValueError(f"Failed to load current_frame.csv: {str(e)}")

# Classify
with torch.no_grad():
    output = model(current_frame)
    _, predicted = torch.max(output, 1)
    disturbance_type = class_labels[predicted.item()]

# Save result
with open('disturbance_type.txt', 'w') as f:
    f.write(disturbance_type)