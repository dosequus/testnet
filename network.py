from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import NUM_ACTIONS

@dataclass
class TakoNetConfig:
    vocab_size: int = 32              # Number of unique FEN tokens
    seq_len: int = 77                 # Fixed length of FEN token sequence
    d_model: int = 128                # Dimensionality of token embeddings and model layers
    num_heads: int = 8               # Number of attention heads in the transformer
    num_layers: int = 15              # Number of transformer encoder layers
    policy_dim: int = NUM_ACTIONS     # Dimensionality of policy head output
    value_dim: int = 3                # Dimensionality of value head output
    dropout: Optional[float] = 0    # Dropout rate for transformer layers

    def create_model(self, device = 'cpu'):
        """
        Creates a TakoNet model using the current configuration.
        
        Returns:
            TakoNet: An instance of the model configured with the current parameters.
        """
        device = torch.device(device)
        return TakoNet(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            policy_dim=self.policy_dim,
            value_dim=self.value_dim,
            dropout=self.dropout,
            device=device
        )
        
class TakoNet(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, num_heads, num_layers, policy_dim, value_dim, dropout, device = 'cpu'):
        super(TakoNet, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)  # Token embedding
        self.positional_encoding = nn.Embedding(seq_len, d_model)  # Learnable positional encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, dropout=dropout),
            num_layers
        ).to(device)
        self.policy_head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, policy_dim)
        ).to(device)
        self.value_head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, value_dim)
        ).to(device)
        
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.TransformerEncoderLayer):
                for submodule in module.children():
                    init_weights(submodule) 
        self.apply(init_weights)
        
    def __call__(self, tokens) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(tokens.to(self.device))

    def forward(self, tokens):
        """
        Args:
            tokens: Tensor of shape [batch_size, seq_len] (FEN tokens as integers).
        
        Returns:
            policy: Tensor of shape [batch_size, 8, 8, 73].
            value: Tensor of shape [batch_size, 3].
        """
        # Token embedding + positional encoding
        batch_size, seq_len = tokens.shape
        position_indices = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        positional_embeds = self.positional_encoding(position_indices)
        x = self.embedding(tokens) + positional_embeds
        # Transformer encoder
        x = self.transformer(x)
        # Flatten sequence for output heads
        x_flat = x.flatten(1)  # Shape: [batch_size, seq_len * d_model]
        # Policy and value predictions
        policy = self.policy_head(x_flat).view(-1, TakoNetConfig.policy_dim)
        value = self.value_head(x_flat).view(-1, TakoNetConfig.value_dim)
        return policy, value
    
    def forward_single(self, tokens: torch.Tensor):
        tokens = tokens.unsqueeze(0)
        p, v = self.forward(tokens.to(self.device))
        return p.squeeze(0), v.squeeze(0)
    
    def predict(self, tokens: torch.Tensor):
        with torch.no_grad():
            return self.forward(tokens.to(self.device))
    
    def predict_single(self, tokens: torch.Tensor):
        with torch.no_grad():
            return self.forward_single(tokens.to(self.device))
    
    def policy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(pred.shape[0], -1)
            
        return nn.CrossEntropyLoss()(pred.to(self.device), target.to(self.device))
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        