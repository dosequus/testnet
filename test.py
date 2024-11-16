import torch
import torch.optim as optim
from model import TransformerNet
import os

checkpoint_path = "checkpoints/best-model.pt" # TODO: configure with command line args
epoch = 0

model = TransformerNet()
optimizer = optim.Adam(model.parameters())

if os.path.isfile(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Checkpoint loaded successfully.")

# print(optimizer.state_dict())
# print(model.state_dict())