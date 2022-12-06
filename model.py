import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from chess import Move, Piece, PieceType, Square
from vit_pytorch import ViT
import numpy as np


class TransformerNet():
    class PositionalEncoding(nn.Module):
        def __init__(self, d_hidden, max_len=64) -> None:
            super().__init__()
            pe = torch.arange(max_len).unsqueeze(1)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(x, mask)

    def loss(self, data_true, data_pred):
        v_true = data_true['value']
        v_pred = data_pred['value']
        return F.mse_loss(v_pred, v_true)

    def predict(self, x: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            return self.forward(x, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.reshape(12, 8, 8).unsqueeze(0)
        print(x.dtype)
        z = self.vit(x)

        z = z.squeeze().reshape(8, 8, -1)
        print(z.shape)
        v = self.value_head(z)
        v = v.reshape(-1).sum()
        v = torch.tanh(v)

        pi = self.policy_head(z)
        pi = pi.reshape(8, 8, -1)
        pi = F.softmax(pi, dim=2)
        pi = pi * mask
        pi = F.softmax(pi, dim=2)
        # return non for possible policy_head implementation
        return pi, v

    @staticmethod
    def get_legal_move_mask(moves: "list[Move]", piece_map: "dict[Square, Piece]") -> torch.TensorType:
        # pi[i][j][k] = Pr(positions[i][j] * t(k) | state)
        KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                        (-2, -1), (-1, -2), (1, -2), (2, -1)]
        # QUEEN_MOVES = [N,NE,E,SE,S,SW,W,NW]
        QUEEN_MOVES = [(0, 1), (1, 1), (0, 1), (-1, 1),
                       (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

        mask = torch.zeros(8, 8, 73)  # moves are illegal to start
        for move in moves:
            from_i, to_i = chess.square_file(
                move.from_square), chess.square_file(move.to_square)
            from_j, to_j = chess.square_rank(
                move.from_square), chess.square_rank(move.to_square)

            if move.promotion is not None and move.promotion != chess.QUEEN:  # "underpromotions
                RIGHT = int(to_i > to_j)
                mask[from_i][from_j][UNDERPROMOTIONS.index(
                    move.promotion)+RIGHT] = 1

            if piece_map[move.from_square].piece_type == chess.KNIGHT:  # knight moves
                di = to_i - from_i
                dj = to_j - from_j

                mask[from_i][from_j][56+KNIGHT_MOVES.index((di, dj))] = 1
            if piece_map[move.from_square].piece_type in (chess.PAWN, chess.KING, chess.ROOK, chess.BISHOP, chess.QUEEN):
                di = to_i - from_i
                dj = to_j - from_j

                di = di//abs(di) if di > 0 else 0
                dj = dj//abs(dj) if dj > 0 else 0

                mask[from_i][from_j][QUEEN_MOVES.index(
                    (di, dj))*7+chess.square_distance(move.from_square, move.to_square)-1] = 1

        return mask
        # if piece_map[move.from_square]

    @staticmethod
    def pi_to_move_map(pi: torch.Tensor, moves: "list[Move]", piece_map: "dict[Square, Piece]") -> "dict[Move, int]":
        KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                        (-2, -1), (-1, -2), (1, -2), (2, -1)]
        # QUEEN_MOVES = [N,NE,E,SE,S,SW,W,NW]
        QUEEN_MOVES = [(0, 1), (1, 1), (0, 1), (-1, 1),
                       (0, -1), (-1, -1), (-1, 0), (-1, 1)]

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

                di = di//abs(di) if di > 0 else 0
                dj = dj//abs(dj) if dj > 0 else 0

                move_mapping[move] = pi[from_i][from_j][QUEEN_MOVES.index(
                    (di, dj))*7+chess.square_distance(move.from_square, move.to_square)-1].item()
        return move_mapping

    # TODO make sure that works lol

    def save_model(self, filepath='checkpoints/best_model'):
        torch.save(self)

    @staticmethod
    def load_model(filepath):
        model = torch.load(filepath)
        return model

    def checkpoint(self):
        return super().checkpoint()
