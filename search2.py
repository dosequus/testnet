import torch
import math
import numpy as np
import tqdm
from collections import defaultdict

from games.chessboard import ChessGame

from model import TransformerNet

def _get_move_mask(game: ChessGame) -> torch.Tensor:
    moves = game.valid_moves()
    pieces = game.game.piece_map()
    return TransformerNet.get_legal_move_mask(moves, pieces)


class Node:
    def __init__(self, parent, state, prior_prob=0):
        self.parent = parent
        self.state = state
        self.children = {}
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.action_val = 0

    def UCB(self):
        return self.prior_prob / (1 + self.visit_count)

    # Will only ever be called on something with children
    def select(self):
        best_move, best_child = max(self.children.items(), key=lambda x: x[1].action_val + x[1].UCB())
        return best_move, best_child 

    def create_child(self, move, val):
        new_state = ChessGame.next_state(self.state, move)
        return Node(self, new_state, prior_prob=val)

    def expand(self, model: TransformerNet):
        game = ChessGame.load(self.state)
        pi, val = model.predict_single(torch.from_numpy(self.state), _get_move_mask(game))
        pred_moves = model.pi_to_move_map(
            pi, game.valid_moves(), game.game.piece_map())
        self.action_val = val
        self.children = {move: self.create_child(move, eval) for move, eval in pred_moves.items()}

    def backup(self):
        # If we reach the root we do nothing
        if self.parent:
            # Otherwise increment parent visit count and adjust their action_val
            new_action_val = self.parent.action_val * self.parent.visit_count
            new_action_val = new_action_val + self.action_val
            self.parent.visit_count = self.parent.visit_count + 1
            self.parent.action_val = new_action_val / self.parent.visit_count
            self.parent.backup()

    def update_pi(self, pi, sim):
        for move, child in self.children.items():
            pi[move] = pi[move] - (pi[move] - child.action_val) / sim


def run(game: ChessGame, model: TransformerNet, num_sim=10, max_depth=50):
    pi = defaultdict(int)
    root = None
    for sim in range(1,num_sim+1):
        # Start MCTS
        root = Node(None, game.state)
        # print(root.children)
        while not game.over():
            # ------------------------------------------------------------------
            #                              SELECT
            # ------------------------------------------------------------------
            # Initialize our depth count and start at root
            depth = 1
            curr_node = root
            # while curr_node is not a leaf
            while len(curr_node.children) > 0:
                _, curr_node = curr_node.select()
                depth = depth + 1
            # print(curr_node)
            # curr_node is now the best leaf node
            # ------------------------------------------------------------------
            #                              EXPAND
            # ------------------------------------------------------------------
            curr_node.expand(model)
            # ------------------------------------------------------------------
            #                              BACKUP
            # ------------------------------------------------------------------
            curr_node.backup()
            if depth == max_depth or root.visit_count > 100:
                # Update pi based on root nodes final values
                root.update_pi(pi, sim)
                break
    return root, max(pi, key=pi.get)
