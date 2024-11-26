import torch
import math
import logging
import numpy as np

from collections import defaultdict

from chess import Move, Board, Outcome, Termination
from games.chessboard import ChessGame
from network import TakoNet
from tokenizer import tokenize
import sys

logger = logging.getLogger(__name__)

class Node:
    
    DRAWING_MOVE_PROB = 0.005
    
    def __init__(self, parent, fen, temperature, prior_prob=0):
        self.parent = parent
        self.fen = fen
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
        if self.temperature > sys.float_info.epsilon:
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

    def create_child(self, board: Board, move: Move, val: int):
        board.push(move)
        node = Node(self, board.fen(), self.temperature*0.9, prior_prob=val)
        if board.outcome() == Outcome(Termination.FIVEFOLD_REPETITION, None):
            node.prior_prob = self.DRAWING_MOVE_PROB
        board.pop()
        return node

    def expand(self, nnet: TakoNet, memo):
        def wdl_to_value(p_win, p_draw, p_loss):
            return p_win - p_loss + 0.5*p_draw
        
        game = ChessGame(self.fen, nnet.device)

        # Check if the current state has already been evaluated and is in memo
        if self.fen in memo:
            self.action_val = memo[self.fen]
            return

        # Get model predictions
        mask = game.get_legal_move_mask()
        policy_logits, value_logits = nnet.predict_single(game.to_tensor())
        # pi, val = 
        value_probs = value_logits.softmax(-1)
        policy_probs = (policy_logits + mask.log()).softmax(-1)
        pred_moves = game.create_move_map(policy_probs)

        curr_score = game.score()
        if curr_score == 1:
            value_probs = torch.tensor([1., 0., 0.], device=value_probs.device)
        elif curr_score == 0:
            value_probs = torch.tensor([0., 1., 0.], device=value_probs.device)
        elif curr_score == -1:
            value_probs = torch.tensor([0., 0., 1.], device=value_probs.device)

        if game.turn == -1:  
            self.action_val = wdl_to_value(*reversed(value_probs.tolist()))
        else: 
            self.action_val = wdl_to_value(*value_probs.tolist())

        memo[self.fen] = self.action_val
        
        # Expand children
        self.children = {move: self.create_child(game.board, move, eval) for move, eval in pred_moves.items()}
        self.children = dict(filter(lambda x:x[1], self.children.items()))

    def backup(self):
        if self.parent:
            new_action_val = self.parent.action_val * self.parent.visit_count
            new_action_val = new_action_val + self.action_val
            self.parent.visit_count += 1
            self.parent.action_val = new_action_val / self.parent.visit_count
            self.parent.backup()
            
    def select_leaf(self):
        if len(self.children) > 0:
            _, child = self.select()
            return child.select_leaf()
        return self

class MCTS:
    def __init__(self, nnet: TakoNet, explore_factor=np.sqrt(2)):
        self.nnet = nnet
        self.memo = {}
        self.explore_factor = explore_factor
        
    def add_dirichlet_noise(self, root: Node, alpha: float = 0.3, epsilon: float = 0.25) -> np.ndarray:
        dirichlet_noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children):
            root.children[child].prior_prob = (1 - epsilon) * root.children[child].prior_prob + epsilon * dirichlet_noise[i]
    
    def run(self, root: Node, num_sim=10, think_time=-1) -> tuple[Node, Move]:
        if think_time > 0:
            import time
            start = time.time()
            while time.time() - start < think_time:
                leaf = root.select_leaf()
                leaf.expand(self.nnet, self.memo)
                leaf.backup()
                self.add_dirichlet_noise(root)
        else:
            for _ in range(num_sim):
                leaf = root.select_leaf()
                leaf.expand(self.nnet, self.memo)
                leaf.backup()
                self.add_dirichlet_noise(root)
        return root, root.select()[0]

# Example usage:
# mcts = MCTS(model)
# root, best_move = mcts.run(game, num_sim=100, max_depth=50)
