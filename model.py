import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from nnet import ViTEncoderOnly
from vit_pytorch import SimpleViT
import numpy as np


def _calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w

class TransformerNet(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        transformer_dim = 64
        seq_len = 64
        self.vit = ViTEncoderOnly(
            image_size=8,
            patch_size=1,
            dim=transformer_dim,
            depth=4,
            heads=8,
            mlp_dim=4*transformer_dim,
            channels=12,
        ).float().to(self.device)
        

        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Linear(transformer_dim // 2, 3),
        ).to(self.device)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                    in_channels=transformer_dim, 
                    out_channels=2, 
                    kernel_size=1, 
                    stride=1, 
                    bias=False
                ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * transformer_dim, 8*8*73),
            nn.Unflatten(1, (8,8))
        ).to(self.device)
        
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        self.apply(init_weights)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def policy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(pred.shape[0], -1)
            
        return nn.CrossEntropyLoss()(pred, target)

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            return self.forward(x.to(self.device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared encoder
        x = x.permute(0, 3, 1, 2)
        z = self.vit(x) # output: [batch_size, seq_len, transformer_dim]
        # Transpose to [batch_size, transformer_dim, sequence_length]
        z = z.permute(0, 2, 1)
        
        # Value head
        value_logits = self.value_head(z)  # Output: [batch_size, 3]
        # Policy head
        policy_logits = self.policy_head(z)  # Output: [batch_size, 8, 8, 73]
        print(policy_logits)
        return policy_logits, value_logits

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Handle a single input by temporarily adding a batch dimension.
        """
        x = x.unsqueeze(0)  # Add batch dimension
        pi, v = self.forward(x.to(self.device))
        return pi.squeeze(0), v.squeeze(0)  # Remove batch dimension
    
    def predict_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward_single(x)

    def save_model(self, filepath='checkpoints/best_model'):
        torch.save({}, filepath)

    @staticmethod
    def load_model(filepath, device='cpu'):
        model = torch.load(filepath, map_location=device)
        return model.to(device)

    def checkpoint(self):
        return super().checkpoint()

