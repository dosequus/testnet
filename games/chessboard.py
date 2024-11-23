
import sys, os, random
from copy import deepcopy
from itertools import product
import numpy as np
import chess
from chess import Move, Piece, PieceType, Square

from functools import cache

from games.games import AbstractGame
import torch

from tokenizer import tokenize, tokenize_action, NUM_ACTIONS


pieces = ["P", "N", "B", "R", "Q", "K"]



class ChessGame(AbstractGame):

    def __init__(self, fen=chess.STARTING_FEN, device = 'cpu'):
        super().__init__()
        self.board = chess.Board(fen)
        self.device = torch.device(device)
        
    @property
    def fen(self):
        return self.board.fen()
        
    @property
    def ply(self):
        return self.board.ply()
    
    @property
    def move_count(self):
        return self.board.fullmove_number
    
    @property
    def turn(self):
        return 1 if self.board.turn else -1 # 1 if white -1 if black

    def score(self):
        if self.board.is_stalemate():
            return 0
        if not self.board.is_game_over():
            return None
        if self.board.outcome().winner is None:
            return 0
        if self.board.outcome().winner:
            return 1 
        return -1

    def valid_moves(self):
        return list(self.board.legal_moves)

    def over(self):
        return self.board.is_game_over()

    def make_move(self, move):
        self.board.push(move)

    def undo_move(self):
        self.board.pop()
    
    def to_tensor(self, device='cpu'):
        return torch.tensor(tokenize(self.fen), dtype=torch.long, device=device)

    def __str__(self):
        return str(self.board)

    def __repr__(self):
        return repr(self.board)  
    
    def get_legal_move_mask(self) -> torch.Tensor:
        mask = torch.zeros(NUM_ACTIONS, device=self.device)  # moves are illegal to start
        for move in self.valid_moves():
            i = tokenize_action(move.uci())
            mask[i] = 1
        return mask + 1e-10
    
    def create_move_map(self, probs: torch.Tensor) -> "dict[Move, int]":
        move_mapping = {}
        for move in self.valid_moves():
            move_mapping[move] = probs[tokenize_action(move.uci())].item()
        return move_mapping
    
    def create_sparse_policy(self, policy: dict[chess.Move, torch.Tensor]):
        policy_tensor = torch.zeros(NUM_ACTIONS, device=self.device)  # moves are illegal to start
        for move in policy:
            policy_tensor[tokenize_action(move.uci())] = policy[move].item()
        return policy_tensor

    def copy(self):
        return ChessGame(self.board.fen(), self.device)