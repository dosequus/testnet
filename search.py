import torch
import math
import logging
import numpy as np

import concurrent.futures

from collections import defaultdict

from chess import Move
from games.chessboard import ChessGame
from model import TransformerNet


logger = logging.getLogger(__name__)


def _get_move_mask(game: ChessGame) -> torch.Tensor:
    moves = game.valid_moves()
    pieces = game.game.piece_map()
    return TransformerNet.get_legal_move_mask(moves, pieces)


class MCTS:
    def __init__(self, nnet: TransformerNet):
        self.nnet = nnet
        self.memo = {}

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

        def select(self):
            """ Select the child node with the highest action value plus UCB. """
            best_move, best_child = max(self.children.items(), key=lambda x: x[1].action_val + x[1].UCB())
            return best_move, best_child

        def create_child(self, move, val):
            new_state = ChessGame.next_state(self.state, move)
            return MCTS.Node(self, new_state, prior_prob=val)

        def expand(self, nnet: TransformerNet, memo):
            game = ChessGame.load(self.state)

            # Check if the current state has already been evaluated and is in memo
            state_key = game.game.fen()  # Use FEN (Forsyth-Edwards Notation) as a unique identifier
            if state_key in memo:
                self.action_val = memo[state_key]
                return

            # Get model predictions
            pi, val = nnet.predict_single(torch.from_numpy(self.state), _get_move_mask(game))
            pred_moves = nnet.pi_to_move_map(pi, game.valid_moves(), game.game.piece_map())

            win_prob, draw_prob, loss_prob = val

            if game.turn == 1:  # White's turn
                self.action_val = win_prob - loss_prob
            else:  # Black's turn
                self.action_val = loss_prob - win_prob

            # If this is a terminal state, memoize the result
            if game.over():
                memo[state_key] = self.action_val

            # Expand children
            self.children = {move: self.create_child(move, eval) for move, eval in pred_moves.items()}

        def backup(self):
            if self.parent:
                new_action_val = self.parent.action_val * self.parent.visit_count
                new_action_val = new_action_val + self.action_val
                self.parent.visit_count += 1
                self.parent.action_val = new_action_val / self.parent.visit_count
                self.parent.backup()

        def update_pi(self, pi, sim):
            for move, child in self.children.items():
                pi[move] = pi[move] - (pi[move] - child.action_val) / sim
                
    def _simulate(self, game, max_depth, max_nodes):
        root = self.Node(None, game.state)
        depth = 0
        curr_node = root

        # Selection and Expansion phase
        while not game.over():
            # SELECT
            while len(curr_node.children) > 0:
                _, curr_node = curr_node.select()
                depth += 1

            # EXPAND
            curr_node.expand(self.nnet, self.memo)

            # BACKUP
            curr_node.backup()

            # Terminate if maximum depth or visit count reached
            if depth == max_depth or root.visit_count > max_nodes:
                break
        return root

    def run(self, game: ChessGame, num_sim=10, max_depth=50, max_nodes=10) -> tuple[Node | None, Move]:
        pi = defaultdict(int)
        roots = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._simulate, game.copy(), max_depth, max_nodes) for _ in range(num_sim)]
            
            for future in concurrent.futures.as_completed(futures):
                root = future.result()
                if root:
                    roots.append(root)
        
        for sim, root in enumerate(roots):
            root.update_pi(pi, sim+1)

        return roots[-1] if roots else None, max(pi, key=pi.get)

# Example usage:
# mcts = MCTS(model)
# root, best_move = mcts.run(game, num_sim=100, max_depth=50)
