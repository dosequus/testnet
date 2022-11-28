import torch
import math
import numpy as np

from games.chessboard import ChessGame

from model import TransformerNet

def score(parent, child):
    prior = child.prior * math.sqrt( parent.prior / (child.visit_count + 1))

    if child.visit_count > 0:
        value = -child.value()
    else: 
        value = 0

    return value + prior

class Node:
    def __init__(self, prior, player) -> None:
        self.visited = 0
        self.player = player
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    @property
    def leaf(self):
        return len(self.children) > 0

    def value(self):
        if self.visited == 0:
            return 0 
        else:
            return self.value_sum / self.visited

    def select_action(self, temperature):
        visit_counts = np.array((child.visited for child in self.children.values()))
        actions = (action for action in self.children.keys())

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float('inf'):
            action = np.random.choice(actions)
        else:
            visited_distribution = visit_counts ** (1/temperature)
            visited_distribution = visited_distribution / np.sum(visited_distribution)
            action = np.random.choice(actions, p=visited_distribution)
        return action
    
    def select_child(self):
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child
    
    def expand(self, state, player, pi):
        self.player = player
        self.state = state
        for a, pr in enumerate(pi):
            if pr != 0:
                self.children[a] = Node(prior=pr, player=(self.player * -1))
    
class MCTS:
    
    def __init__(self, game : ChessGame, **kwargs) -> None:
        self.game = game
        self.num_simulations = kwargs.get('num_simulations')

    def best_move(self, model : TransformerNet):
        root = self.run(self.game, model, self.game.turn)
        return root.select_action(0)


    def run(self, game : ChessGame, model: TransformerNet, player, depth=50):
        root = Node(0, player)

        state = game.state()
        valid_moves = game.valid_moves()
        pi = np.array(( (mv, math.pow(len(valid_moves), -1)) for mv in valid_moves))
        
        root.expand(state, player=player, pi=pi)

        for _ in range(self.num_simulations):
            a, node = None, root
            stack = [node]
            curr_depth = depth

            while not node.leaf:
                a, node = node.select_child()
                stack.append(node)
                curr_depth -= 1

            # we are at a leaf node
            # state = n
            
            # make a move
            next_state = game.next_state(state, a)
            value = game.state_score(player)

            if value is None: 
                pi, value = model.predict(next_state, torch.empty(1))
                # node.expand(next_state, player=)

            # self.backup(stack, value, parent.player * -1)
            if curr_depth == 0: break

        return root 

    def backup(self, stack : list[Node], value, player):
        while stack:
            node = stack.pop(-1)
            node.value_sum += (value if node.player else -value)
            node.visited += 1