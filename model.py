import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from chess import Move, Piece, PieceType, Square
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
            nn.Linear(64, 1),
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
        v = v.reshape(batch_size, -1).sum(dim=1)
        v = torch.tanh(v)

        # Forward pass through the policy head
        pi = self.policy_head(z)
        pi = pi.reshape(batch_size, 8, 8, -1)
        pi = F.softmax(pi, dim=-1)

        # Apply the mask
        pi = pi * mask.to(self.device)
        pi = F.softmax(pi, dim=-1)
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

    @staticmethod
    def get_legal_move_mask(moves: "list[Move]", piece_map: "dict[Square, Piece]", device='cpu') -> torch.TensorType:
        # pi[i][j][k] = Pr(positions[i][j] * t(k) | state)
        KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                        (-2, -1), (-1, -2), (1, -2), (2, -1)]
        # QUEEN_MOVES = [N,NE,E,SE,S,SW,W,NW]
        QUEEN_MOVES = [(0, 1), (1, 1), (1, 0), (-1, 1),
                       (0, -1), (-1, -1), (-1, 0), (1, -1)]

        UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
        
        KNIGHT_PLANE = 56
        UNDERPROMOTE_PLANE = KNIGHT_PLANE+8

        mask = torch.zeros(8, 8, 73, device=device)  # moves are illegal to start
        for move in moves:
            from_i, to_i = chess.square_file(
                move.from_square), chess.square_file(move.to_square)
            from_j, to_j = chess.square_rank(
                move.from_square), chess.square_rank(move.to_square)
            
            piece_type = piece_map[move.from_square].piece_type

            if move.promotion is not None and move.promotion != chess.QUEEN:  # "underpromotions
                RIGHT = int(to_i > to_j)
                mask[from_i][from_j][UNDERPROMOTE_PLANE+UNDERPROMOTIONS.index(
                    move.promotion)+RIGHT] = 1

            if piece_type == chess.KNIGHT:  # knight moves
                di = to_i - from_i
                dj = to_j - from_j

                mask[from_i][from_j][56+KNIGHT_MOVES.index((di, dj))] = 1
            if piece_type in (chess.ROOK, chess.BISHOP, chess.QUEEN, ):
                di = to_i - from_i
                dj = to_j - from_j
                
                di = di//abs(di) if abs(di) > 0 else 0
                dj = dj//abs(dj) if abs(dj) > 0 else 0

                if (di, dj) in QUEEN_MOVES:
                    distance = max(abs(to_i - from_i), abs(to_j - from_j))
                    direction_index = QUEEN_MOVES.index((di, dj))
                    plane_index = direction_index * 7 + (distance - 1)
                    mask[from_i][from_j][plane_index] = 1
                else:
                    print("ruh roh: ", (di, dj))
                    exit(1)
        return mask
    
    @staticmethod
    def pi_to_move_map(pi: torch.Tensor, moves: "list[Move]", piece_map: "dict[Square, Piece]") -> "dict[Move, int]":
        KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                        (-2, -1), (-1, -2), (1, -2), (2, -1)]
        # QUEEN_MOVES = [N,NE,E,SE,S,SW,W,NW]
        QUEEN_MOVES = [(0, 1), (1, 1), (1, 0), (-1, 1),
                       (0, -1), (-1, -1), (-1, 0), (1, -1)]

        UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
        move_mapping = {}
        for move in moves:
            from_i, to_i = chess.square_file(
                move.from_square), chess.square_file(move.to_square)
            from_j, to_j = chess.square_rank(
                move.from_square), chess.square_rank(move.to_square)

            if move.promotion is not None and move.promotion != chess.QUEEN:  # "underpromotions
                RIGHT = int(to_i > to_j)
                move_mapping[move] = pi[from_i][from_j][UNDERPROMOTIONS.index(
                    move.promotion)+RIGHT].item()

            if piece_map[move.from_square].piece_type == chess.KNIGHT:  # knight moves
                di = to_i - from_i
                dj = to_j - from_j

                move_mapping[move] = pi[from_i][from_j][56 +
                                                        KNIGHT_MOVES.index((di, dj))].item()
            if piece_map[move.from_square].piece_type in (chess.PAWN, chess.KING, chess.ROOK, chess.BISHOP, chess.QUEEN):
                di = to_i - from_i
                dj = to_j - from_j

                di = di//abs(di) if abs(di) > 0 else 0
                dj = dj//abs(dj) if abs(dj) > 0 else 0

                move_mapping[move] = pi[from_i][from_j][QUEEN_MOVES.index(
                    (di, dj))*7+chess.square_distance(move.from_square, move.to_square)-1].item()
        return move_mapping

    def save_model(self, filepath='checkpoints/best_model'):
        torch.save({}, filepath)

    @staticmethod
    def load_model(filepath, device='cpu'):
        model = torch.load(filepath, map_location=device)
        return model.to(device)

    def checkpoint(self):
        return super().checkpoint()

