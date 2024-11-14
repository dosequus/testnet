
import sys, os, random
from copy import deepcopy
from itertools import product
import numpy as np
import chess
from chess import Move, Piece, PieceType, Square

from functools import cache

from games.games import AbstractGame
import torch


pieces = ["P", "N", "B", "R", "Q", "K"]



class ChessGame(AbstractGame):

    def __init__(self, fen=chess.STARTING_FEN):
        super().__init__()
        self.board = chess.Board(fen)
        
    @property
    def ply(self):
        return self.board.ply()
    
    @property
    def turn(self):
        return 1 if self.board.turn else -1 # 1 if white -1 if black
    
    @property
    def state(self):
        return self.board
    
    @property
    def move_count(self):
        return self.board.fullmove_number

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
        
    @classmethod
    def next_state(cls, state : chess.Board, move: Move):
        next_state = state.copy(stack=False)
        next_state.push(move)
        return next_state
    
    @classmethod
    def load(cls, state):
        game = cls()
        game.board = state
        return game

    # @cache()
    def to_numpy(self):
        H, W = 8,8
        
        state = np.zeros((H, W, len(pieces)+6), dtype=np.float32)
        
        for square in range(64):
            i = chess.square_file(square)
            j = chess.square_rank(square)
            piece = self.board.piece_at(square)
            if piece:
                # White pieces are +1, black pieces are -1
                state[i, j, piece.piece_type - 1] = +1 if piece.color else -1

        KW, KB = self.board.has_kingside_castling_rights(chess.WHITE), self.board.has_kingside_castling_rights(chess.BLACK)
        QW, QB = self.board.has_queenside_castling_rights(chess.WHITE), self.board.has_queenside_castling_rights(chess.BLACK)
        state[:, :, len(pieces)].fill(self.turn)
        state[:, :, len(pieces)+1].fill(self.ply)
        state[:, :, len(pieces)+2].fill(1 if KW else 0)
        state[:, :, len(pieces)+3].fill(1 if QW else 0)
        state[:, :, len(pieces)+4].fill(1 if KB else 0)
        state[:, :, len(pieces)+5].fill(1 if QB else 0)

        return state
    
    def to_tensor(self, device='cpu'):
        return torch.from_numpy(self.to_numpy()).to(device)

    @classmethod
    def from_numpy(cls, state : np.ndarray):
        # piece_map = {} # piece letter => k value
        
                       
        game = cls(fen=None)
        indices = np.argwhere(state[:,:,:len(pieces)] != 0)
        for i, j, k in indices:
            square = chess.square(i, j)
            game.board.set_piece_at(square, chess.Piece(k+1, state[i,j,k] == 1))
        # game.board.set_piece_map(piece_map)
        
        game.board.turn = state[0, 0, len(pieces)] > 0
        game.ply = int(state[0, 0, len(pieces) + 1])
        game.board.fullmove_number = (game.ply // 2) + 1
        
        castling = ""
        if state[0,0,len(pieces)+2]: castling += 'K'
        if state[0,0,len(pieces)+3]: castling += 'Q'
        if state[0,0,len(pieces)+4]: castling += 'k'
        if state[0,0,len(pieces)+5]: castling += 'q'
        game.board.set_castling_fen(castling)

        return game
                    




    def __str__(self):
        return str(self.board)

    def __repr__(self):
        return repr(self.board)
    
    
    def get_legal_move_mask(self, device='cpu') -> torch.TensorType:
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
        for move in self.valid_moves():
            from_i, to_i = chess.square_file(
                move.from_square), chess.square_file(move.to_square)
            from_j, to_j = chess.square_rank(
                move.from_square), chess.square_rank(move.to_square)
            
            piece_type = self.board.piece_type_at(move.from_square)

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
    
    def pi_to_move_map(self, pi: torch.Tensor) -> "dict[Move, int]":
        KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                        (-2, -1), (-1, -2), (1, -2), (2, -1)]
        # QUEEN_MOVES = [N,NE,E,SE,S,SW,W,NW]
        QUEEN_MOVES = [(0, 1), (1, 1), (1, 0), (-1, 1),
                       (0, -1), (-1, -1), (-1, 0), (1, -1)]

        UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
        move_mapping = {}
        for move in self.valid_moves():
            from_i, to_i = chess.square_file(
                move.from_square), chess.square_file(move.to_square)
            from_j, to_j = chess.square_rank(
                move.from_square), chess.square_rank(move.to_square)

            if move.promotion is not None and move.promotion != chess.QUEEN:  # "underpromotions
                RIGHT = int(to_i > to_j)
                move_mapping[move] = pi[from_i][from_j][UNDERPROMOTIONS.index(
                    move.promotion)+RIGHT].item()

            piece_type = self.board.piece_type_at(move.from_square)
            
            if piece_type == chess.KNIGHT:  # knight moves
                di = to_i - from_i
                dj = to_j - from_j

                move_mapping[move] = pi[from_i][from_j][56 +
                                                        KNIGHT_MOVES.index((di, dj))].item()
            if piece_type in (chess.PAWN, chess.KING, chess.ROOK, chess.BISHOP, chess.QUEEN):
                di = to_i - from_i
                dj = to_j - from_j

                di = di//abs(di) if abs(di) > 0 else 0
                dj = dj//abs(dj) if abs(dj) > 0 else 0

                move_mapping[move] = pi[from_i][from_j][QUEEN_MOVES.index(
                    (di, dj))*7+chess.square_distance(move.from_square, move.to_square)-1].item()
        return move_mapping
    
    def pi_to_policy(self, policy: dict[chess.Move, torch.Tensor]):
        
        # pi[i][j][k] = Pr(positions[i][j] * t(k) | state)
        KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                        (-2, -1), (-1, -2), (1, -2), (2, -1)]
        # QUEEN_MOVES = [N,NE,E,SE,S,SW,W,NW]
        QUEEN_MOVES = [(0, 1), (1, 1), (1, 0), (-1, 1),
                       (0, -1), (-1, -1), (-1, 0), (1, -1)]

        UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
        
        KNIGHT_PLANE = 56
        UNDERPROMOTE_PLANE = KNIGHT_PLANE+8

        policy_tensor = torch.zeros(8, 8, 73)  # moves are illegal to start
        for move in policy:
            from_i, to_i = chess.square_file(
                move.from_square), chess.square_file(move.to_square)
            from_j, to_j = chess.square_rank(
                move.from_square), chess.square_rank(move.to_square)
            
            piece_type = self.board.piece_type_at(move.from_square)

            if move.promotion is not None and move.promotion != chess.QUEEN:  # "underpromotions
                RIGHT = int(to_i > to_j)
                policy_tensor[from_i][from_j][UNDERPROMOTE_PLANE+UNDERPROMOTIONS.index(
                    move.promotion)+RIGHT] = policy[move]

            if piece_type == chess.KNIGHT:  # knight moves
                di = to_i - from_i
                dj = to_j - from_j

                policy_tensor[from_i][from_j][56+KNIGHT_MOVES.index((di, dj))] = policy[move]
            if piece_type in (chess.ROOK, chess.BISHOP, chess.QUEEN, ):
                di = to_i - from_i
                dj = to_j - from_j
                
                di = di//abs(di) if abs(di) > 0 else 0
                dj = dj//abs(dj) if abs(dj) > 0 else 0

                if (di, dj) in QUEEN_MOVES:
                    distance = max(abs(to_i - from_i), abs(to_j - from_j))
                    direction_index = QUEEN_MOVES.index((di, dj))
                    plane_index = direction_index * 7 + (distance - 1)
                    policy_tensor[from_i][from_j][plane_index] = policy[move]
                else:
                    print("ruh roh: ", (di, dj))
                    exit(1)
        return policy_tensor




if __name__ == "__main__":
    
    game = ChessGame()
    game = ChessGame.load(game.state())
    print (game)
    print (repr(game))
    print ()

    for i in range(0, 100):
        actions = list(game.valid_moves())
        game.make_move(random.choice(actions))

        game = ChessGame.load(game.state())

        print (game)
        print (repr(game))
        print ()

