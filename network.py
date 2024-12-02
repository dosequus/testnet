from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
from tokenizer import NUM_ACTIONS, SEQUENCE_LENGTH, VOCAB_SIZE, _CHARACTERS


class PositionalEncodings(enum.Enum):
    SINUSOID = enum.auto()
    LEARNED = enum.auto()


@dataclass
class TakoNetConfig:
    vocab_size: int = VOCAB_SIZE        # Number of unique FEN tokens
    seq_len: int = SEQUENCE_LENGTH      # Fixed length of FEN token sequence
    d_model: int = 64                  # Dimensionality of token embeddings and model layers
    num_heads: int = 4                  # Number of attention heads in the transformer
    num_layers: int = 4                 # Number of transformer encoder layers
    policy_dim: int = NUM_ACTIONS       # Dimensionality of policy head output
    value_dim: int = 3                  # Dimensionality of value head output
    dropout: Optional[float] = 0      # Dropout rate for transformer layers
    widening_factor: int = 4
    pos_encodings: PositionalEncodings = PositionalEncodings.LEARNED

    def create_model(self, device = 'cpu'):
        """
        Creates a TakoNet model using the current configuration.
        
        Returns:
            TakoNet: An instance of the model configured with the current parameters.
        """
        device = torch.device(device)
        return TakoNet(
            config=self,
            device=device
        )
        
        
class TakoNet(nn.Module):
    
    class SwiGLU(nn.Module):
        def __init__(self, w1, w2, w3):
            super().__init__()
            self.w1
            self.w2
            self.w3
        
        def forward(self, x):
            x1 = F.linear(x, self.w1.weight)
            x2 = F.linear(x, self.w2.weight)
            hidden = F.silu(x1) * x2
            return F.linear(hidden, self.w3.weight)

        def __call__(self, x):
            return self.forward(x)
        
    def __init__(self, 
                 config: TakoNetConfig,
                 device = 'cpu'):
        super(TakoNet, self).__init__()
        self.device = device
        self.seq_len = config.seq_len + 2
        self.sos_token = _CHARACTERS.index('SOS')
        self.eos_token = _CHARACTERS.index('EOS')
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, device=device)  # Token embedding
        match config.pos_encodings:
            case PositionalEncodings.LEARNED:
                self.positional_encoding = nn.Embedding(self.seq_len, config.d_model, device=device)  # Learnable positional encoding
            case PositionalEncodings.SINUSOID:
                self.positional_encoding = self.sinusoid_position_encoding(
                    sequence_length = self.seq_len, 
                    hidden_size = config.d_model,
                )
                
        ffn_dim = config.widening_factor * config.d_model
                
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(config.d_model, 
                                       config.num_heads, 
                                       dim_feedforward=ffn_dim, 
                                       batch_first=True,
                                       dropout=config.dropout),
            config.num_layers
        ).to(device)
        self.layer_norm = nn.LayerNorm(config.d_model, device=self.device)
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model, config.policy_dim),
            # nn.GELU(),
            # nn.Linear(d_model, policy_dim)
        ).to(device)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.value_dim),
            # nn.GELU(),
            # nn.Linear(d_model, value_dim)
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
            policy: Tensor of shape [batch_size, 1968].
            value: Tensor of shape [batch_size, 3].
        """
        # Token embedding + positional encoding
        tokens = self.add_special_tokens(tokens)
        batch_size, seq_len = tokens.shape
        position_indices = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        positional_embeds = self.positional_encoding(position_indices)
        x = self.embedding(tokens) + positional_embeds
        # Transformer decoder
        x = self.transformer(x, torch.zeros_like(x))
        normalized_output = self.layer_norm(x)
        # pooled_output = normalized_output.mean(dim=1)
        last_token_output = normalized_output[:, -1, :]
        # Policy and value predictions
        policy = self.policy_head(last_token_output)
        value = self.value_head(last_token_output)
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
    
    def sinusoid_position_encoding(self,
        sequence_length: int,
        hidden_size: int,
        max_timescale: float = 1e4,
    ):
        pe = torch.zeros(sequence_length, hidden_size)
        position = torch.arange(0, sequence_length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, hidden_size, 2, dtype=torch.float) *
                            -(math.log(max_timescale) / hidden_size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return lambda _: pe
    
    def add_special_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Adds SOS and EOS tokens to the input sequence.
        Args:
            tokens (torch.Tensor): Input token sequences of shape [batch_size, seq_len].
        Returns:
            torch.Tensor: Sequences with SOS and EOS tokens added, shape [batch_size, seq_len + 2].
        """
        batch_size = tokens.size(0)
        sos_tokens = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=tokens.device)
        eos_tokens = torch.full((batch_size, 1), self.eos_token, dtype=torch.long, device=tokens.device)
        return torch.cat([sos_tokens, tokens, eos_tokens], dim=1)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        