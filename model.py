import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from chess import Move, is_legal, Piece, PieceType, Square
from vit_pytorch import ViT
import numpy as np

class TransformerNet():
    class PositionalEncoding(nn.Module):
        def __init__(self, d_hidden, max_len=64) -> None:
            super().__init__()
            pe = torch.arange(max_len).unsqueeze(1)
            self.register_buffer('pe', pe)

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:x.size(0)]
            return x
    """
        ntoken: size of tokens => 12 for chess
        emsize = 
    """
    def __init__(self):
        super().__init__()
        self.vit = ViT(
            image_size=8,
            patch_size=1,
            num_classes=1024,
            dim=256,
            depth=2,
            heads=1,
            mlp_dim=256,
            channels=12,
            # pool='mean'
        ).float()
        self.value_head = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # TODO implement policy head
        self.policy_head = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 73),
        )

    def __call__(self, x : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        return self.forward(x, mask)

    def loss(self, data_true, data_pred):
        v_true = data_true['value']
        v_pred = data_pred['value']
        return F.mse_loss(v_pred, v_true)

    def predict(self, x : torch.Tensor, mask : torch.Tensor):
        with torch.no_grad():
            return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(12, 8, 8).unsqueeze(0)
        print(x.dtype)
        z = self.vit(x)

        z = z.squeeze().reshape(8,8,-1)
        print(z.shape)
        v = self.value_head(z)
        v = v.reshape(-1).sum()
        v = torch.tanh(v)

        pi = self.policy_head(z)
        pi = pi.reshape(8,8,-1)
        pi = F.softmax(pi, dim=1)
        # return non for possible policy_head implementation
        return pi, v

    def get_legal_move_mask(moves: list[Move], piece_map: dict[Square, Piece]):
        # pi[i][j][k] = Pr(positions[i][j] * t(k) | state)
        knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

        mask = torch.zeros(8, 8, 73) # moves are illegal to start
        for move in moves:
            from_i, to_i = chess.square_file(move.from_square), chess.square_file(move.to_square)
            from_j, to_j =  chess.square_rank(move.from_square), chess.square_rank(move.to_square)
            if piece_map[move.from_square].piece_type == chess.KNIGHT:
                di = to_i - from_i
                dj = to_j - from_j

                mask[from_i][from_j][knight_moves.index((di,dj))] = 1

            if piece_map[move.from_square]
                


    # TODO make sure that works lol
    def save_model(self, filepath='checkpoints/best_model'):
        torch.save(self)

    @staticmethod
    def load_model(filepath):
        model = torch.load(filepath)
        return model
        
    def checkpoint(self):
        return super().checkpoint()