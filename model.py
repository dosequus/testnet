import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
import numpy as np

class TransformerNet(nn.Module):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_hidden, max_len=64, device='cpu') -> None:
            super().__init__()
            pe = torch.arange(max_len).unsqueeze(1).to(device)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:x.size(0)]
            return x

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        self.vit = ViT(
            image_size=8,
            patch_size=1,
            num_classes=1024,
            dim=256,
            depth=2,
            heads=1,
            mlp_dim=256,
            channels=12,
        ).float().to(self.device)

        self.value_head = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        ).to(self.device)
        
        self.policy_head = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 73),
        ).to(self.device)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(x, mask)

    def loss(self, data_true, data_pred):
        v_true = data_true['value'].to(self.device)
        v_pred = data_pred['value'].to(self.device)
        return F.mse_loss(v_pred, v_true)

    def predict(self, x: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            return self.forward(x.to(self.device), mask.to(self.device))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)  # Get the batch size

        # Reshape x to fit the input shape expected by ViT
        x = x.reshape(batch_size, 12, 8, 8).to(self.device)
        z = self.vit(x)  # ViT should now handle batch inputs correctly

        z = z.squeeze()  # Remove unnecessary dimensions
        z = z.reshape(batch_size, 8, 8, -1)  # Reshape to [batch_size, 8, 8, hidden_dim]

        # Forward pass through the value head
        v = self.value_head(z)
        v = v.reshape(batch_size, -1, 3).sum(dim=1)
        v = F.softmax(v, dim=-1)
        # Forward pass through the policy head
        pi = self.policy_head(z)
        pi = pi.reshape(batch_size, 8, 8, -1)
        pi = F.softmax(pi, dim=-1)

        # Apply the mask
        pi = pi * mask.to(self.device)
        # pi = F.softmax(pi, dim=-1)
        return pi, v

    def forward_single(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Handle a single input by temporarily adding a batch dimension.
        """
        x = x.unsqueeze(0)  # Add batch dimension
        mask = mask.unsqueeze(0)  # Add batch dimension
        pi, v = self.forward(x.to(self.device), mask.to(self.device))
        return pi.squeeze(0), v.squeeze(0)  # Remove batch dimension
    
    def predict_single(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward_single(x.to(self.device), mask.to(self.device))

    def save_model(self, filepath='checkpoints/best_model'):
        torch.save({}, filepath)

    @staticmethod
    def load_model(filepath, device='cpu'):
        model = torch.load(filepath, map_location=device)
        return model.to(device)

    def checkpoint(self):
        return super().checkpoint()

