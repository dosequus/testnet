import torch
import math
import numpy as np

from games.chessboard import ChessGame

from model import TransformerNet


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
        best_move, best = next(iter(self.children))
        for move, child in self.children:
            if (child.action_val + child.UCB() > best.action_val + best.UCB()):
                best_move, best_child = move, child
        return best_move, best_child

    def create_child(self, move, eval):
        game = ChessGame.load(self.state)
        game.make_move(move)
        return Node(self, game.state, eval)

    def expand(self, model: TransformerNet, game: ChessGame):
        pi, val = model.predict(self.state, torch.empty(1))
        self.children = model.pi_to_move_map(
            pi, game.valid_moves, game.game.piece_map())
        self.action_val = val
        self.children = {move: self.create_child(
            self, move, eval) for move, eval in self.children.items()}

    def backup(self):
        # If we reach the root we do nothing
        if self.parent:
            # Otherwise increment parent visit count and adjust their action_val
            new_action_val = self.parent.action_val * self.parent.visit_count
            new_action_val = new_action_val + self.action_val
            self.parent.visit_count = self.parent.visit_count + 1
            self.parent.action_val = new_action_val / self.parent.visit_count
            self.parent.backup()


def run(game: ChessGame, model: TransformerNet, num_sim=10, max_depth=50):
    for _ in range(num_sim):
        root = Node(None, game.state())
        # Start MCTS
        while True:
            # ----------------------------------------------------------------------
            #                                  SELECT
            # ----------------------------------------------------------------------
            # Initialize our depth count and start at root
            depth = 0
            curr_node = root
            # while curr_node is not a leaf
            while len(curr_node.children) > 0:
                _, curr_node = curr_node.select()
                depth = depth + 1
            # curr_node is now the best leaf node
            # ----------------------------------------------------------------------
            #                                  EXPAND
            # ----------------------------------------------------------------------
            curr_node.expand(model, game)
            # ----------------------------------------------------------------------
            #                                  BACKUP
            # ----------------------------------------------------------------------
            curr_node.backup()
            if depth == max_depth:
                # At this point we want to do something with the root node
                # presumably save the weights of its children or smth?
                break
            depth = 0
