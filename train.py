import torch
import torch.backends
import torch.nn as nn
from games.chessboard import ChessGame
from model import TransformerNet
import torch.optim as optim
import search
import random
from settings import Configuration
import tqdm
import time
import concurrent.futures
from collections import deque
import os
import evaluate
from chessboard import display

config = Configuration().get_config()
class Arena:
    def __init__(self, model, total_games, batch_size, device='cpu'):
        """
        Initialize the Arena.
        
        :param model_path: Path to the model that will be loaded for all agents.
        :param total_games: Total number of self-play games to conduct.
        :param batch_size: The batch size for compiling states for training.
        :param device: The device to run the model on (e.g., 'cpu' or 'cuda').
        """
        self.model = model
        self.total_games = total_games
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=config.training.replay_buffer_size)  # Adjust size based on expected memory usage


    def pit_agents(self):
        """Pit the agents against each other and collect the game data."""
        temp_memory = list()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.self_play_game, i) for i in range(self.total_games)]
            w, d, l = 0, 0, 0
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    states, policies, masks, target_result = future.result()
                    if torch.equal(target_result, torch.tensor([1., 0., 0.])):
                        w += 1
                    elif torch.equal(target_result, torch.tensor([0., 1., 0.])):
                        d += 1
                    else:
                        l += 1  
                    targets = [target_result]*len(states)
                    temp_memory.extend(list(zip(states, policies, masks, targets)))
                    print((w+d+l), "/", self.total_games, " games completed", " | added ", len(states), " new states to replay buffer", sep='')
                else:
                    print("Game tossed for error")
            self.memory.extend(temp_memory)
            print(f"Arena record: +{w}={d}-{l}")

    def self_play_game(self, t):
        """Conduct a self-play game with a given MCTS agent and return the results."""
        try:
            
            mcts_agent = search.MCTS(self.model)  # Each process gets its own MCTS instance
            game = ChessGame()
            states, policies, masks = [], [], []
            # board_window = None if not config.visualize else display.start()
            while not game.over():
                root, best_move = mcts_agent.run(game, 
                                                max_depth=config.training.max_depth, 
                                                num_sim=config.training.num_simulations, 
                                                max_nodes=config.mcts.max_nodes)
                
                total_visits = 1 + sum(child.visit_count for child in root.children.values())
                policy_map = {move: torch.tensor(node.visit_count / total_visits) for move, node in root.children.items()}
                
                states.append(game.to_tensor())
                policies.append(game.pi_to_policy(policy_map).reshape(config.model.policy_output_size))
                masks.append(game.get_legal_move_mask())
                
                game.make_move(best_move)
                # if config.visualize: display.update(game.board.fen(), board_window)  
            
            
            result = game.score()
            if result == 1:
                target_result = torch.tensor([1.0, 0.0, 0.0])  # Win
            elif result == 0:
                target_result = torch.tensor([0.0, 1.0, 0.0])  # Draw
            else:
                target_result = torch.tensor([0.0, 0.0, 1.0])  # Loss
            
            return states, policies, masks, target_result
        except Exception as e:
            # toss the game
            print(e)
            return None

    def compile_batch(self):
        """Compile a batch of states for training."""
        if len(self.memory) >= self.batch_size:
            batch = random.sample(list(self.memory), self.batch_size)
            state_batch, policy_batch, mask_batch, value_batch = zip(*batch)
            
            state_tensor = torch.stack(state_batch).to(self.device)
            policy_tensor = torch.stack(policy_batch).to(self.device)
            mask_tensor = torch.stack(mask_batch).to(self.device)
            value_tensor = torch.stack(value_batch).to(self.device)
            
            return state_tensor, policy_tensor, mask_tensor, value_tensor
        else:
            return None

# Example Usage of Arena in Training
def train(model: TransformerNet, optimizer: torch.optim.Optimizer, device='cpu', starting_epoch=0):
    arena = Arena(model=model, 
                  total_games=config.training.num_self_play_games, 
                  batch_size=config.training.batch_size, 
                  device=device)
    
    for epoch in range(starting_epoch, config.training.num_epochs):
        print(f"Epoch: {epoch+1}")
        print("Creating training games through arena play...")
        arena.pit_agents()
        print("Beginning training steps...")
        for _ in tqdm.trange(config.training.training_steps):
            batch = arena.compile_batch()
            if batch:
                state_tensor, policy_tensor, mask_tensor, value_tensor = batch
                
                # Forward pass
                predicted_policies, predicted_values = model(state_tensor, mask_tensor)
                
                # Compute loss
                predicted_policies = predicted_policies.reshape(config.training.batch_size, config.model.policy_output_size)
                policy_loss = nn.CrossEntropyLoss()(predicted_policies, policy_tensor)
                value_loss = nn.MSELoss()(predicted_values, value_tensor)
                total_loss = policy_loss + value_loss
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        print(f"Total loss at epoch {epoch+1}: {total_loss.item()}")
        with open("loss.txt", 'a') as logfile:
            logfile.write(f'{epoch+1},{total_loss.item()}\n')
        if (epoch+1) % config.training.evaluation_interval == 0:
            board_window = None if not config.visualize else display.start()
            
            evaluate.stockfish_benchmark(search.MCTS(model, explore_factor=0), num_games=config.evaluation.num_games, device=device, game_board=board_window)
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f'checkpoints/best-model.pt') 
        # print(total_loss)

if __name__ == '__main__':
    # Determine the device to use: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu') 
    print(device.type)
    
    model = TransformerNet()
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)
    checkpoint_path = "checkpoints/best-model.pt" # TODO: configure with command line args
    epoch = 0
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
    
    print(config)
    
    train(model, optimizer, starting_epoch=epoch)