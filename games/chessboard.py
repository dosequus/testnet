
import sys, os, random
from copy import deepcopy
from itertools import product
import numpy as np
import chess

from games.games import AbstractGame
import torch
import IPython


pieces = ["P", "N", "B", "R", "Q", "K"]



class ChessGame(AbstractGame):

    def __init__(self, fen=chess.Board().fen()):
        super().__init__()
        self.game = chess.Board(fen)

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

    def state_score(self, state, player=1):
        g = ChessGame.load(state)
        score = g.score()
        if score is None: return None
        if score == 0: return 0
        return 1 if score == player else -1

    def valid_moves(self):
        moves = []
        for mv in self.game.legal_moves:
            self.game.push(mv)
            if self.game.is_valid():
                moves.append(mv)
            self.game.pop()
        return moves

    def next_state(self, state, move):
        g = ChessGame.load(state)
        g.push(move)
        state = g.state()
        return state

    def over(self):
        return self.game.is_game_over()

    def make_move(self, move):
        self.game.push(move)

    def undo_move(self):
        self.game.pop()

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
        state[:, :, len(pieces)+1].fill(self.game.ply())
        state[:, :, len(pieces)+2].fill(1 if KW else 0)
        state[:, :, len(pieces)+3].fill(1 if QW else 0)
        state[:, :, len(pieces)+4].fill(1 if KB else 0)
        state[:, :, len(pieces)+5].fill(1 if QB else 0)

        return state 

    def __str__(self):
        return str(self.game)

    def __repr__(self):
        return repr(self.game)




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


