import torch
import math
import logging
import numpy as np

import concurrent.futures

from collections import defaultdict

from chess import Move
from games.chessboard import ChessGame
from model import TransformerNet
import time

logger = logging.getLogger(__name__)

class MCTS:
    def __init__(self, nnet: TransformerNet, explore_factor=np.sqrt(2)):
        self.nnet = nnet
        self.memo = {}
        self.explore_factor = explore_factor
        
        self.nnet.share_memory()

    class Node:
        def __init__(self, parent, state, temperature, prior_prob=0):
            self.parent = parent
            self.state = state
            self.temperature = temperature
            self.children = {}
            self.prior_prob = prior_prob
            self.visit_count = 0
            self.action_val = 0
            
        def UCB(self):
            return self.prior_prob / (1 + self.visit_count)

        def select(self):
            """ 
            Select the child node using a temperature parameter to balance exploration and exploitation. 

            temperature = 1.0: Balanced exploration and exploitation.
            temperature = 0.1: More deterministic selection, favoring the best scores.
            temperature = 0.0: Purely deterministic selection, equivalent to always choosing the highest score
        
            """
            # Get the action values and UCB scores for each child
            scores = np.array([child.action_val + child.UCB() for child in self.children.values()])
            
            # Apply the temperature to the scores
            if self.temperature > 0:
                scaled_scores = scores / self.temperature
            else:
                scaled_scores = scores

            # Calculate probabilities using softmax
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Subtract max for numerical stability
            probabilities = exp_scores / np.sum(exp_scores)

            # Choose a child node based on the probabilities
            chosen_index = np.random.choice(len(self.children), p=probabilities)
            
            # Get the corresponding move and child node
            best_move = list(self.children.keys())[chosen_index]
            best_child = list(self.children.values())[chosen_index]
            
            return best_move, best_child

        def create_child(self, move, val):
            new_state = ChessGame.next_state(self.state, move)
            return MCTS.Node(self, new_state, self.temperature*.70, prior_prob=val)

        def expand(self, nnet: TransformerNet, memo):
            
            game = ChessGame.load(self.state)

            # Check if the current state has already been evaluated and is in memo
            state_key = game.board.fen()  # Use FEN (Forsyth-Edwards Notation) as a unique identifier
            if state_key in memo:
                self.action_val = memo[state_key]
                return

            # Get model predictions
            pi, val = nnet.predict_single(game.to_tensor(), game.get_legal_move_mask())
            # pi, val = (torch.rand(8, 8, 73), torch.rand(3))
            pred_moves = game.pi_to_move_map(pi)

            curr_score = game.score()
            if curr_score == 1:
                val = torch.tensor([1., 0., 0.]).to(val.device)
            elif curr_score == 0:
                val = torch.tensor([0., 1., 0.]).to(val.device)
            elif curr_score == -1:
                val = torch.tensor([0., 0., 1.]).to(val.device)

            win_prob, draw_prob, loss_prob = val.tolist()

            if game.turn == -1:  # white just played a move
                # Prioritize wins > draws > losses
                self.action_val = win_prob + draw_prob / 2 - loss_prob
            else:  # Black's turn
                self.action_val = loss_prob + draw_prob / 2 - win_prob

            # If this is a terminal state, memoize the result
            if game.score():
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
        root = self.Node(None, game.state, self.explore_factor)
        # Selection and Expansion phase
        while not game.over():
            curr_node = root
            depth = 0
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
