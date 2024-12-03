import torch
import torch.backends
from games.chessboard import ChessGame
import torch.optim as optim
import torch.nn as nn
from settings import Configuration
import tqdm
import pandas as pd
import os
import argparse
import zstandard as zstd
import requests
import random
import evaluate
from chess import Move
from network import TakoNet, TakoNetConfig
from stockfish import Stockfish
from math import ceil
from sklearn.metrics import f1_score
from torch.utils.tensorboard.writer import SummaryWriter
import tables
import numpy as np
from dataset import PuzzleDataset
from torch.utils.data.dataloader import DataLoader


config = Configuration().get_config()


def download_training_set():
    """Download puzzles from Lichess and extract them to a CSV file."""
    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    compressed_path = "./puzzles/lichess_db_puzzle.csv.zst"
    extracted_path = "./puzzles/lichess_db_puzzle.csv"

    os.makedirs("./puzzles", exist_ok=True)

    if not os.path.exists(extracted_path):
        print("Downloading puzzle dataset...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 8192
            with open(compressed_path, 'wb') as f, tqdm.tqdm(total=total_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("Download complete. Extracting dataset...")

            with open(compressed_path, 'rb') as compressed_file, open(extracted_path, 'wb') as extracted_file:
                decomp = zstd.ZstdDecompressor()
                decomp.copy_stream(compressed_file, extracted_file)

        print("Extraction complete.")

    else:
        print("Puzzle dataset already exists. Skipping download and extraction.")
        


def preprocess_row(row):
        fen = row["FEN"]
        moves = row["Moves"].split()
        game = ChessGame(fen)
        
        first_move, best_move = moves[:2]
        
        game.make_move(Move.from_uci(first_move))
        # State tensor
        state_tensor = game.to_tensor()
        # Policy tensor
        pi = {Move.from_uci(best_move): torch.tensor(1.0)}
        policy_tensor = game.create_sparse_policy(pi)
        
        #mask
        mask = game.get_legal_move_mask()
        # Value tensor placeholder (replace with actual WDL if available)
        wdl = [.6, .2, .2] if game.turn == 1 else [.2, .2, .6]
        value_tensor = torch.tensor(wdl, dtype=torch.float)

        return state_tensor, mask, policy_tensor, value_tensor, torch.tensor(row['Rating'])

def preprocess_data(puzzle_csv_path: str, num_puzzles: int, save_dir: str = "puzzles"):
    """
    Preprocess the entire chess puzzles dataset into tensors, split into training and validation sets,
    and save them to disk.
    Args:
        puzzle_csv_path (str): Path to the CSV file containing puzzles.
        save_dir (str): Directory to save the preprocessed tensors.
    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "data.h5")

    # Check if already preprocessed
    if os.path.isfile(save_path):
        print("Already pre-processed.")
        return;

    # Load the CSV into a DataFrame
    puzzles = pd.read_csv(puzzle_csv_path, nrows=num_puzzles, usecols=['FEN', 'Moves', 'Rating'])
    print(f"Loaded {len(puzzles)} puzzles from {puzzle_csv_path}.")

    with tables.open_file(save_path, mode='w') as h5file:
        filters = tables.Filters(complevel=5)
    
        state_storage = h5file.create_earray(h5file.root, "states", tables.IntAtom(), shape=(0,TakoNetConfig.seq_len), filters=filters)
        mask_storage = h5file.create_earray(h5file.root, "masks", tables.FloatAtom(), shape=(0,TakoNetConfig.policy_dim), filters=filters)
        policy_storage = h5file.create_earray(h5file.root, "policies", tables.FloatAtom(), shape=(0,TakoNetConfig.policy_dim), filters=filters)
        value_storage = h5file.create_earray(h5file.root, "values", tables.FloatAtom(), shape=(0,TakoNetConfig.value_dim), filters=filters)
        rating_storage = h5file.create_earray(h5file.root, "ratings", tables.IntAtom(), shape=(0,1), filters=filters)

        # Preprocess rows and save to HDF5
        for _, row in tqdm.tqdm(puzzles.iterrows(), total=len(puzzles)):
            state, mask, policy, value, rating = preprocess_row(row)

            state_storage.append(state.unsqueeze(0).numpy())
            mask_storage.append(mask.unsqueeze(0).numpy())
            policy_storage.append(policy.unsqueeze(0).numpy())
            value_storage.append(value.unsqueeze(0).numpy())
            rating_storage.append(rating.reshape((1,1)).numpy())
        
        print(f"Preprocessed data saved to {save_path}")


def get_data_loaders(hdf5_path='puzzles/data.h5', train_ratio=0.8, num_workers=4):
    """
    Create DataLoader objects for training and validation.

    Args:
        hdf5_path (str): Path to the HDF5 file.
        batch_size (int): Number of samples per batch.
        train_ratio (float): Ratio of data used for training.
        num_workers (int): Number of workers for data loading.
    """
    train_dataset = PuzzleDataset(hdf5_path, split="train", train_ratio=train_ratio)
    val_dataset = PuzzleDataset(hdf5_path, split="val", train_ratio=train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=config.pretrain.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.pretrain.validation_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def calculate_f1_score(logits, truth):
    # Convert policy logits to predicted classes
    predicted_classes = torch.argmax(logits, dim=1)
    
    # Convert true_policy to true classes (indices of one-hot vector)
    true_classes = torch.argmax(truth, dim=1)
    
    # Compute F1 score
    f1 = f1_score(true_classes.cpu().numpy(), predicted_classes.cpu().numpy(), average="macro")
    
    return f1

def calculate_top_k_accuracy(logits, truth, k = 3):

     # Convert truth to class indices if it's one-hot encoded
    if truth.ndim > 1 and truth.size(1) > 1:
        truth = torch.argmax(truth, dim=1)  # Shape: [batch_size]
    # Get the indices of the top-k predictions
    top_k_predictions = torch.topk(logits, k=k, dim=1).indices  # Shape: [batch_size, k]

    # Check if the true class is in the top-k predictions
    correct = (top_k_predictions == truth.unsqueeze(1))  # Shape: [batch_size, k]
    # Compute the Top-k Accuracy
    top_k_accuracy = correct.any(dim=1).float().mean().item()  # Average across the batch

    return top_k_accuracy

def train(model: TakoNet, optimizer: torch.optim.Optimizer, starting_epoch=0):    
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.pretrain.num_epochs, 
        eta_min=1e-6, 
        verbose=True
    )
    
    alpha = config.pretrain.alpha
    best_rating = 0
    writer = SummaryWriter(log_dir="./logs/pretraining")
    train_loader, val_loader = get_data_loaders()
    for epoch in range(starting_epoch, config.pretrain.num_epochs):
        print("Epoch:", epoch+1)
        pbar = tqdm.tqdm(train_loader)
        for batch in pbar:
            state_tensor, mask, true_policy, true_value, ratings = map(lambda x: x.to(model.device), batch)
            # ratings = ratings.squeeze(0)
            # Forward pass
            policy_logits, value_logits = model(state_tensor)
            # Compute loss
            value_loss = nn.CrossEntropyLoss(label_smoothing=0.05)(value_logits, true_value)
            policy_loss = nn.CrossEntropyLoss(label_smoothing=0.05)(policy_logits + mask.log(), true_policy)
            total_loss = (1 - alpha) * value_loss + alpha * policy_loss
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            accuracy = calculate_top_k_accuracy(policy_logits+mask.log(), true_policy, k=1)
            score = round(accuracy * policy_logits.size(0))
            puzzle_rating = evaluate.performance_rating(ratings.flatten().tolist(), score)
            pbar.set_description_str(f"loss={total_loss.item():.3f}, elo={puzzle_rating}, ACC={accuracy:.3g}")
        
            writer.add_scalar('Loss/Train', total_loss.item())
            writer.add_scalar('Accuracy/Train', accuracy)
            writer.add_scalar('Elo/Train', puzzle_rating)
        
        # validation
        if (epoch+1) % config.pretrain.validation_interval == 0:
            all_policy_logits = []
            all_true_policies = []
            all_masks = []
            all_ratings = []
            val_loss = 0
            pbar = tqdm.tqdm(val_loader)
            for batch in pbar:
                state_tensor, mask, true_policy, true_value, ratings = map(lambda x: x.to(device), batch)
                
                policy_logits, value_logits = model.predict(state_tensor)
                
                value_loss = nn.CrossEntropyLoss(label_smoothing=0.05)(value_logits, true_value)
                policy_loss = nn.CrossEntropyLoss(label_smoothing=0.05)(policy_logits + mask.log(), true_policy)
                val_loss += ((1 - alpha) * value_loss + alpha * policy_loss).item() 

                all_policy_logits.append(policy_logits)
                all_true_policies.append(true_policy)
                all_masks.append(mask)
                all_ratings.append(ratings)
                
            
                
            all_policy_logits = torch.cat(all_policy_logits, dim=0)
            all_true_policies = torch.cat(all_true_policies, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            all_ratings = torch.cat(all_ratings, dim=0)


            # Apply the mask to logits (ensuring illegal moves have minimal contribution)
            masked_logits = all_policy_logits + all_masks.log()

            # Calculate top-k accuracy over the entire validation set
            accuracy = calculate_top_k_accuracy(masked_logits, all_true_policies, k=1)

            # Calculate total score for puzzle performance
            score = round(accuracy * all_policy_logits.size(0))

            # Estimate puzzle rating from performance
            model_puzzle_rating = evaluate.performance_rating(
                all_ratings.flatten().tolist(), 
                score
            )
            writer.add_scalar('Loss/Val', val_loss)
            writer.add_scalar('Accuracy/Val', accuracy)
            writer.add_scalar('Elo/Val', model_puzzle_rating)
            print(f"val puzzle score: {score}, Estimated puzzle rating: {model_puzzle_rating}")
            if model_puzzle_rating > best_rating:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'policy_loss': total_loss,
                    'puzzle_rating': model_puzzle_rating, 
                }, f'checkpoints/best-pretrained-model.pt')
                print("model saved.")
                best_rating = model_puzzle_rating
        # scheduler step
        scheduler.step()        
    writer.close()
    
def choose_device():
    # Determine the device to use: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu') 
    return device
    
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pretrain tako chess model and manage checkpoints.")
    parser.add_argument(
        '--device',
        default=choose_device(),
        help='Device to run the model on (default: auto)'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='small',
        help='Configure size of the model (default: small)'
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/best-pretrained-model.pt",
        help="Path to save or load model checkpoints (default: checkpoints/best-pretrained-model.pt)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/final_model.pt",
        help="Path to save the final trained model (default: models/final_model.pt)"
    )
    parser.add_argument(
        "--num_puzzles",
        type=int,
        default=None,
        help="Number of puzzles to train on (default: all)"
    )
    parser.add_argument(
        "--puzzle_path",
        type=str,
        default="puzzles/lichess_db_puzzle.csv",
        help="Path to puzzle dataset (default: puzzles/lichess_db_puzzle.csv)"
    )
    args = parser.parse_args()
    
    model_size = args.model_size   
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    device = args.device
    num_puzzles = args.device
    csv_path = args.puzzle_path
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Create model
    model = TakoNetConfig(model_size=model_size).create_model(device) # pass device to create_model for GPU
    print(f"Loaded {model_size} with {model.count_params():,} params")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
        
    # download dataset
    download_training_set()
    print("Loading puzzles to memory...")
    preprocess_data(csv_path, num_puzzles)
    
    print("Training the model...")
    train(model, optimizer, starting_epoch=0)

    print(f"Saving final model to: {save_path}")
    # (Add logic to save the final model)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, f'{save_path}/{model.get_canonical_name()}.pt') 

if __name__ == '__main__':
    print(config)
    main()