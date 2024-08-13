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
import evaluate

config = Configuration().get_config()

def train(model: TransformerNet, optimizer: optim.Optimizer, device='cpu'):
    
    memory = deque(maxlen=config.training.replay_buffer_size)
    total_loss = torch.tensor(0)
    
    mcts = search.MCTS(model)
    
    for epoch in range(config.training.num_epochs):
        print(f"Epoch: {epoch+1}")
        # self play
        for _ in tqdm.trange(config.training.num_self_play_games):
            game = ChessGame()
            
            states = []
            policies = []
            masks = []
            
            while not game.over():
                # print("\n===============")
                # print(game.game)
                # print("===============")
                root, best_move = mcts.run(game.copy(), 
                                           max_depth=config.training.max_depth, 
                                           num_sim=config.training.num_simulations, 
                                           max_nodes=config.mcts.max_nodes)
                
                total_visits = sum(child.visit_count for child in root.children.values())
                policy_map = { move : torch.tensor(node.visit_count/total_visits) for move, node in root.children.items() }
                
                states.append(torch.tensor(game.state, dtype=torch.float32))
                policies.append(game.pi_to_policy(policy_map).reshape(config.model.policy_output_size))
                masks.append(TransformerNet.get_legal_move_mask(policy_map.keys(), game.game.piece_map()))
                
                game.make_move(best_move)
            
            result = game.score()
            
            # Create the target result tensor based on the outcome
            if result == 1:  # Win
                target_result = torch.tensor([1.0, 0.0, 0.0])  # [win, draw, loss]
            elif result == 0:  # Draw
                target_result = torch.tensor([0.0, 1.0, 0.0])  
            else:  # Loss
                target_result = torch.tensor([0.0, 0.0, 1.0]) 
            
            for s, p, m in zip(states, policies, masks):
                memory.append((s, p, m, target_result))

                
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
                
                    
        if (epoch+1) % 10 == 0:  # Evaluate every 10 iterations
            evaluate.stockfish_benchmark(mcts, device=device)
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
    # print(torch.backends.mps.)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(device.type)
    model = TransformerNet(device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)
    train(model, optimizer, device)