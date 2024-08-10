
import sys, os, random
from copy import deepcopy
from itertools import product
import numpy as np
import chess

from games.games import AbstractGame
import torch
import IPython


pieces = ["P", "K", "B", "R", "Q", "N"]



class ChessGame(AbstractGame):

    def __init__(self, fen=chess.Board().fen()):
        super().__init__()
        self.game = chess.Board(fen)
        self.ply = self.game.ply()

    @property
    def turn(self):
        return 1 if self.game.turn else -1 # 1 if white -1 if black

    def score(self):
        if self.game.can_claim_draw() or self.game.is_stalemate():
            return 0
        if not self.game.is_game_over():
            return None
        if self.game.outcome().winner is None:
            return 0
        if self.game.outcome().winner:
            return 1 
        return -1

    @staticmethod
    def state_score(state):
        g = ChessGame.load(state)
        score = g.score()
        if score is None: return None
        if score == 0: return 0
        return 1 if score == g.turn else -1

    def valid_moves(self):
        return list(self.game.legal_moves)

    def over(self):
        return self.game.is_game_over()

    def make_move(self, move):
        self.game.push(move)

    def undo_move(self):
        self.game.pop()
        
    @classmethod
    def next_state(self, state, move):
        g = ChessGame.load(state)
        # print(g.game.fen())
        
        g.make_move(move)
        # print("New state: ", g.game.fen())
        if(g.state is None):
            print(state, self.game.fen(), g.fen())
        return g.state

    @property
    def state(self):
        H, W = 8,8
        
        state = np.zeros((H, W, len(pieces)+6), dtype=np.float32)
        for i, j in product(range(W), range(H)):
            piece = self.game.piece_at(chess.square(i, j))
            if piece:
                # White pieces are +1, black pieces are -1
                state[i, j, pieces.index(piece.symbol().upper())] = +1 if piece.symbol().isupper() else -1

        KW, KB = self.game.has_kingside_castling_rights(chess.WHITE), self.game.has_kingside_castling_rights(chess.BLACK)
        QW, QB = self.game.has_queenside_castling_rights(chess.WHITE), self.game.has_queenside_castling_rights(chess.BLACK)
        state[:, :, len(pieces)].fill(self.turn)
        state[:, :, len(pieces)+1].fill(self.ply)
        state[:, :, len(pieces)+2].fill(1 if KW else 0)
        state[:, :, len(pieces)+3].fill(1 if QW else 0)
        state[:, :, len(pieces)+4].fill(1 if KB else 0)
        state[:, :, len(pieces)+5].fill(1 if QB else 0)

        return state

    @classmethod
    def load(cls, state : np.ndarray):
        piece_map = {} # piece letter => k value
        H, W = 8,8
        for k in range(len(pieces)):
            for i, j in product(range(H), range(W)):
                if state[i, j, k] == 1:
                    piece_map[chess.square(i,j)] = chess.Piece.from_symbol(pieces[k])
                if state[i, j, k] == -1:
                    piece_map[chess.square(i,j)] = chess.Piece.from_symbol(pieces[k].lower())    
        game = ChessGame()
        game.game.set_piece_map(piece_map)
        
        game.game.turn = 1 if state[0,0, len(pieces)] > 0 else 0
        game.game.fullmove_number = int((state[0,0,len(pieces)+1] + 1) // 2)
        game.ply = state[0,0,len(pieces)+1] + int(not(bool(game.game.turn)))
        castling = ""
        if state[0,0,len(pieces)+2]: castling += 'K'
        if state[0,0,len(pieces)+3]: castling += 'Q'
        if state[0,0,len(pieces)+4]: castling += 'k'
        if state[0,0,len(pieces)+5]: castling += 'q'
        game.game.set_castling_fen(castling)

        return game
                    




    def __str__(self):
        return str(self.game)

    def __repr__(self):
        return repr(self.game)
    
    def pi_to_policy(self, policy: dict[chess.Move, torch.Tensor]):
        piece_map = self.game.piece_map()
        
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
            
            piece_type = piece_map[move.from_square].piece_type

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

