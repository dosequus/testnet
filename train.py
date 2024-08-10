import torch
import torch.backends
import torch.nn as nn
from games.chessboard import ChessGame
from model import TransformerNet
import torch.optim as optim
import search2
import random
from settings import load_config
import tqdm
from collections import deque

config = load_config()

def evaluate_and_update_benchmark(new_model, previous_model_path, num_games=10, device='cpu', save_path='checkpoints/best_model', max_depth=5, num_sim=100):
    """
    Evaluate the new model by playing a series of games against the previously saved benchmark model.
    If the new model wins more games, it becomes the new benchmark and is saved.
    
    Args:
    - new_model (nn.Module): The new model to be evaluated.
    - previous_model_path (str): Path to the previously saved benchmark model.
    - num_games (int): Number of games to play between the models.
    - device (str): Device to run the models on ('cpu' or 'cuda').
    - save_path (str): Path to save the new benchmark model if it wins.
    - max_depth (int): Maximum search depth for the MCTS.
    - num_sim (int): Number of simulations for the MCTS.
    
    Returns:
    - bool: True if the new model becomes the new benchmark, False otherwise.
    """

    def play_game(model1, model2):
        """
        Simulate a game between two models using the provided ChessGame and search2 logic.
        Returns 1 if model1 wins, 0 if it's a draw, and -1 if model2 wins.
        """
        game = ChessGame()
        
        while not game.over():
            if game.current_player() == 'white':  # model1 plays as white
                _, best_move = search2.run(game, model1, max_depth=max_depth, num_sim=num_sim)
            else:  # model2 plays as black
                _, best_move = search2.run(game, model2, max_depth=max_depth, num_sim=num_sim)
            
            game.make_move(best_move)
        
        result = game.score()
        return result  # 1 if model1 wins, 0 if draw, -1 if model2 wins

    # Load the previous model
    previous_model = torch.load(previous_model_path, map_location=device)
    previous_model.to(device)
    new_model.to(device)

    new_model_wins = 0
    previous_model_wins = 0
    draws = 0

    for game in range(num_games):
        # Alternate who plays as white/black
        if game % 2 == 0:
            result = play_game(new_model, previous_model)
        else:
            result = play_game(previous_model, new_model)
            result = -result  # Invert the result because the perspective is switched

        if result == 1:
            new_model_wins += 1
        elif result == -1:
            previous_model_wins += 1
        else:
            draws += 1

    print(f"New Model Wins: {new_model_wins}, Previous Model Wins: {previous_model_wins}, Draws: {draws}")

    if new_model_wins > previous_model_wins:
        print("New model is stronger and will replace the previous benchmark.")
        torch.save(new_model, save_path)  # Save new model as the benchmark
        return True
    else:
        print("Previous model remains the benchmark.")
        return False

def train(model: TransformerNet, optimizer: optim.Optimizer, device='cpu'):
    
    memory = deque(maxlen=config.training.replay_buffer_size)
    total_loss = torch.tensor(0)
    for epoch in range(config.training.num_epochs):
        print(f"Epoch: {epoch+1}")
        # self play
        for g in tqdm.trange(config.training.num_self_play_games):
            game = ChessGame("5k2/1R3p2/1p2r2p/8/5pPP/5K2/8/8 b - - 0 38")
            
            states = []
            policies = []
            masks = []
            
            while not game.over():
                # print(game.game)
                root, best_move = search2.run(game, model, 
                                                    max_depth=config.training.max_depth, 
                                                    num_sim=config.training.num_simulations)
                
                total_visits = sum(child.visit_count for child in root.children.values())
                policy_map = { move : torch.tensor(node.visit_count/total_visits) for move, node in root.children.items() }
                
                states.append(torch.tensor(game.state, dtype=torch.float32))
                policies.append(game.pi_to_policy(policy_map).reshape(config.model.policy_output_size))
                masks.append(TransformerNet.get_legal_move_mask(policy_map.keys(), game.game.piece_map()))
                
                game.make_move(best_move)
            
            result = game.score()
            # print(game.game)
            # print("result: ", result)
            for s, p, m in zip(states, policies, masks):
                memory.append((s, p, m, torch.tensor(result, dtype=torch.float32)))
                
        if len(memory) >= config.training.batch_size:
            batch = random.sample(memory, config.training.batch_size)
            state_batch, policy_batch, mask_batch, value_batch = zip(*batch)
            
            # Convert batch lists to tensors
            state_tensor = torch.stack(state_batch).to(device)  # Shape: [batch_size, 12, 8, 8]
            policy_tensor = torch.stack(policy_batch).to(device)  # Shape: [batch_size, 8, 8, 73]
            mask_tensor = torch.stack(mask_batch).to(device)  # Shape: [batch_size, 8, 8]
            value_tensor = torch.tensor(value_batch).to(device)  # Shape: [batch_size]
            
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
                
                    
        if (epoch) % 2 == 0:  # Evaluate every 10 iterations
            # evaluate_model(model)
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f'checkpoints/model{epoch}.pt') 
        # print(total_loss)

if __name__ == '__main__':
    # print(torch.backends.mps.)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(device.type)
    model = TransformerNet(device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)
    train(model, optimizer, device)