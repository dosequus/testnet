import torch
import math
import logging
import numpy as np

from collections import defaultdict

from chess import Move, Board
from games.chessboard import ChessGame
from network import TakoNet
from tokenizer import tokenize

logger = logging.getLogger(__name__)

def _adjust_max_nodes(move_count: int, max_nodes: int, growth_rate: float = 0.02, min_nodes: int = 10) -> int:
    """
    Adjusts the maximum node count for a search algorithm based on the current move count.
    
    Args:
        move_count (int): The current number of moves in the game.
        max_nodes (int): The absolute maximum number of nodes to search.
        growth_rate (float): Controls the rate at which the node count grows with moves.
        min_nodes (int): The minimum number of nodes to search in the early game.

    Returns:
        int: The adjusted maximum node count.
    """
    
    # Calculate the adjustment factor using an exponential/logistic growth model.
    adjustment_factor = 1 - math.exp(-growth_rate * move_count)
    
    # Scale the adjustment factor to fit between min_nodes and max_nodes.
    adjusted_nodes = min_nodes + (max_nodes - min_nodes) * adjustment_factor
    
    return int(adjusted_nodes)

class MCTS:
    def __init__(self, nnet: TakoNet, explore_factor=np.sqrt(2)):
        self.nnet = nnet
        self.memo = {}
        self.explore_factor = explore_factor
        
    class Node:
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

        def create_child(self, board: Board, move: Move, val: int):
            try:
                board.push(move)
                node = MCTS.Node(self, board.fen(), self.temperature*.70, prior_prob=val)
                board.pop()
                return node
            except:
                return None

        def expand(self, nnet: TakoNet, memo):
            
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
            policy_probs = (policy_logits + mask).softmax(-1)
            pred_moves = game.create_move_map(policy_probs)

            curr_score = game.score()
            if curr_score == 1:
                value_probs = torch.tensor([1., 0., 0.], device=value_probs.device)
            elif curr_score == 0:
                value_probs = torch.tensor([0., 1., 0.], device=value_probs.device)
            elif curr_score == -1:
                value_probs = torch.tensor([0., 0., 1.], device=value_probs.device)

            win_prob, draw_prob, loss_prob = value_probs.tolist()

            if game.turn == -1:  # white just played a move
                # Prioritize wins > draws > losses
                self.action_val = win_prob + draw_prob / 2 - loss_prob
            else:  # Black's turn
                self.action_val = loss_prob + draw_prob / 2 - win_prob

            # If this is a terminal state, memoize the result 
            if game.score():
                memo[self.fen] = self.action_val * 100

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

        def update_pi(self, pi, sim):
            for move, child in self.children.items():
                pi[move] = pi[move] - (pi[move] - child.action_val) / sim
    
    def _simulate(self, game, max_depth, max_nodes):
        temperature = self.explore_factor * (0.70**game.move_count)
        root = self.Node(None, game.board.fen(), temperature)
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
        max_nodes = _adjust_max_nodes(game.move_count-1, max_nodes, min_nodes=100)
        for sim in range(num_sim):
            root = self._simulate(game.copy(), max_depth, max_nodes)
            root.update_pi(pi, sim+1)

        return root, max(pi, key=pi.get)

# Example usage:
# mcts = MCTS(model)
# root, best_move = mcts.run(game, num_sim=100, max_depth=50)
