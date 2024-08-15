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
from collections import deque
import os
import evaluate
from chessboard import display

config = Configuration().get_config()

def train(model: TransformerNet, optimizer: optim.Optimizer, device='cpu', starting_epoch=0):
    game_board = None
    if config.visualize: game_board = display.start()
    
    memory = deque(maxlen=config.training.replay_buffer_size)
    total_loss = torch.tensor(0)
    
    mcts = search.MCTS(model)
    
    for epoch in range(starting_epoch, config.training.num_epochs):
        print(f"Epoch: {epoch+1}")
        w,d,l = 0,0,0
        # self play
        pbar = tqdm.trange(config.training.num_self_play_games, desc=f'+{w}={d}-{l}')
        for _ in pbar:
            game = ChessGame()
            
            states = []
            policies = []
            masks = []
            
            while not game.over():
                if config.visualize: display.update(game.game.fen(), game_board)

                root, best_move = mcts.run(game.copy(), 
                                           max_depth=config.training.max_depth, 
                                           num_sim=config.training.num_simulations, 
                                           max_nodes=config.mcts.max_nodes)
                
                total_visits = 1+sum(child.visit_count for child in root.children.values())
                policy_map = { move : torch.tensor(node.visit_count/total_visits) for move, node in root.children.items() }
                
                states.append(torch.tensor(game.state, dtype=torch.float32))
                policies.append(game.pi_to_policy(policy_map).reshape(config.model.policy_output_size))
                masks.append(TransformerNet.get_legal_move_mask(policy_map.keys(), game.game.piece_map()))
                
                game.make_move(best_move)
            
            if config.visualize: display.update(game.game.fen(), game_board)  
            result = game.score()
            # Create the target result tensor based on the outcome
            if result == 1:  # Win
                if config.visualize: display.update(game.game.fen(), game_board)
                if config.verbose: print("white won")
                w += 1
                target_result = torch.tensor([1.0, 0.0, 0.0])  # [win, draw, loss]
            elif result == 0:  # Draw
                if game.game.is_fifty_moves():
                    if config.verbose: print("draw due to fifty moves without capture")
                elif game.game.is_insufficient_material():
                    if config.verbose: print("draw due to insufficient material")
                elif game.game.is_fivefold_repetition():
                    if config.verbose: print("draw due to 5-fold-repetition")
                d += 1
                target_result = torch.tensor([0.0, 1.0, 0.0])  
            else:  # Loss
                if config.visualize: display.update(game.game.fen(), game_board)
                if config.verbose: print("black won")
                l += 1
                target_result = torch.tensor([0.0, 0.0, 1.0]) 
            pbar.set_description_str(f'+{w}={d}-{l}')
            for s, p, m in zip(states, policies, masks):
                memory.append((s, p, m, target_result))

        for _ in tqdm.trange(config.training.training_steps):
            if len(memory) >= config.training.batch_size:
                batch = random.sample(memory, config.training.batch_size)
                state_batch, policy_batch, mask_batch, value_batch = zip(*batch)
                
                # Convert batch lists to tensors
                state_tensor = torch.stack(state_batch).to(device)  # Shape: [batch_size, 12, 8, 8]
                policy_tensor = torch.stack(policy_batch).to(device)  # Shape: [batch_size, 8, 8, 73]
                mask_tensor = torch.stack(mask_batch).to(device)  # Shape: [batch_size, 8, 8]
                value_tensor = torch.stack(value_batch).to(device)  # Shape: [batch_size]
                
                # Forward pass
                predicted_policies, predicted_values = model(state_tensor, mask_tensor)
                # Compute loss
                predicted_policies = predicted_policies.reshape(config.training.batch_size, config.model.policy_output_size)
                # print(predicted_policies.shape, policy_tensor.shape)

                policy_loss = nn.CrossEntropyLoss()(predicted_policies, policy_tensor)
                # print(predicted_values.shape, value_tensor.shape)
                value_loss = nn.MSELoss()(predicted_values, value_tensor)
                # print(policy_loss.shape, value_loss.shape)
                
                total_loss = policy_loss + value_loss
                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
        print(f"total loss: {total_loss.item()}") 
                
                    
        if (epoch+1) % config.training.evaluation_interval == 0:  # Evaluate every 10 iterations
            evaluate.stockfish_benchmark(mcts, device=device, game_board=game_board)
            # Save the model
            stamp = str(int(time.time()))
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
    
    model = TransformerNet(device=device)
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
    
    
    train(model, optimizer, device, epoch)