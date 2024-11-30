import torch
import torch.backends
from games.chessboard import ChessGame
from model import TransformerNet
import torch.optim as optim
import torch.nn as nn
from settings import Configuration
import tqdm
import pandas as pd
import os
import zstandard as zstd
import requests
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import evaluate
from chess import Move
from network import TakoNet, TakoNetConfig
from tokenizer import tokenize
from stockfish import Stockfish
from math import ceil
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

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
    train_save_path = os.path.join(save_dir, "train_chess_puzzles.pt")
    val_save_path = os.path.join(save_dir, "val_chess_puzzles.pt")

    # Check if already preprocessed
    if os.path.isfile(train_save_path) and os.path.isfile(val_save_path):
        print("Already pre-processed.")
        return num_puzzles*.8, num_puzzles*.2

    # Load the CSV into a DataFrame
    puzzles = pd.read_csv(puzzle_csv_path, nrows=num_puzzles, usecols=['FEN', 'Moves', 'Rating'])
    print(f"Loaded {len(puzzles)} puzzles from {puzzle_csv_path}.")

    # Preprocess the dataset in parallel
    results = []
    with ProcessPoolExecutor() as executor, tqdm.tqdm(total=len(puzzles)) as pbar:
        futures = list(executor.submit(preprocess_row, row) for _, row in puzzles.iterrows())

        for f in as_completed(futures):
            pbar.update()
            results.append(f.result())

    # Unpack results
    state_tensors, mask, policy_tensors, value_tensors, ratings = zip(*results)

    # Combine tensors into datasets
    dataset = {
        "state_tensors": torch.stack(state_tensors),
        "mask_tensors": torch.stack(mask),
        "policy_tensors": torch.stack(policy_tensors),
        "value_tensors": torch.stack(value_tensors),
        "rating_tensors": torch.stack(ratings)
    }

    # Split into training and validation sets
    train_indices, val_indices = train_test_split(
        range(len(state_tensors)), test_size=0.2, random_state=42
    )
    train_set = {key: value[train_indices] for key, value in dataset.items()}
    val_set = {key: value[val_indices] for key, value in dataset.items()}

    # Save the datasets to disk
    torch.save(train_set, train_save_path)
    torch.save(val_set, val_save_path)

    print(f"Training data saved to {train_save_path}")
    print(f"Validation data saved to {val_save_path}")
    return len(train_set), len(val_set)


def compile_batches(batch_size: int, preprocessed_path: str = "puzzles/train_chess_puzzles.pt", device = 'cpu'):
    """
    Load preprocessed data and compile a random batch for training.
    Args:
        batch_size (int): Number of samples per batch.
        preprocessed_path (str): Path to the preprocessed tensors file.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batched state, policy, and value tensors.
    """
    # Load preprocessed data
    data = torch.load(preprocessed_path, map_location=device)
    state_tensors = data["state_tensors"]
    policy_tensors = data["policy_tensors"]
    value_tensors = data["value_tensors"]
    mask_tensors = data["mask_tensors"]
    rating_tensors = data["rating_tensors"]

    # Ensure all tensors have the same length
    dataset_size = len(state_tensors)
    assert all(len(tensor) == dataset_size for tensor in [policy_tensors, value_tensors, mask_tensors]), \
        "Mismatch in tensor lengths in preprocessed data."

    # Randomly select indices for the batch
    indices = random.sample(range(dataset_size), batch_size)
    
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    # Create batches
    def batch_generator():
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield (
                state_tensors[batch_indices],
                mask_tensors[batch_indices],
                policy_tensors[batch_indices],
                value_tensors[batch_indices],
                rating_tensors[batch_indices]
            )
        yield None
    
    # Return batched tensors
    return batch_generator()

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

def train(model: TakoNet, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, starting_epoch=0):    
    csv_path = "./puzzles/lichess_db_puzzle.csv"
    print("Loading puzzles to memory...")
    NUM_PUZZLES = 10_000
    len_train, len_val = preprocess_data(csv_path, num_puzzles=NUM_PUZZLES)
    print("Starting pre-training...")
    alpha = config.pretrain.alpha
    validation_set = compile_batches(config.pretrain.validation_batch_size,
                                           "puzzles/val_chess_puzzles.pt")
    for epoch in range(starting_epoch, config.pretrain.num_epochs):
        print("Epoch:", epoch+1)
        batch_count = ceil(len_train/config.pretrain.batch_size) - 1 # subtract 1 for None yield
        pbar = tqdm.tqdm(compile_batches(config.pretrain.batch_size), total=batch_count)
        for batch in pbar:
            if not batch: continue
            state_tensor, mask, true_policy, true_value, _ = batch
            # Forward pass
            policy_logits, value_logits = model(state_tensor)
            # Compute loss
            value_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(value_logits, true_value)
            policy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(policy_logits + mask.log(), true_policy)
            total_loss = (1 - alpha) * value_loss + alpha * policy_loss
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            f1 = calculate_f1_score(policy_logits+mask.log(), true_policy)
            topk = calculate_top_k_accuracy(policy_logits+mask.log(), true_policy)
            pbar.set_description_str(f"loss={total_loss.item():.3f}, T@3={topk:.3g}, F1={f1:.3g}")
        
        with open("logs/pretraining/policy.csv", 'a+') as logfile:
                logfile.write(f"{epoch},{policy_loss.item()}\n")
        with open("logs/pretraining/value.csv", 'a+') as logfile:
                logfile.write(f"{epoch},{value_loss.item()}\n")
        scheduler.step()
        
        if (epoch+1) % config.pretrain.validation_interval == 0:
            batch = next(validation_set)
            if not batch:
                validation_set = compile_batches(config.pretrain.validation_batch_size,
                                           "puzzles/val_chess_puzzles.pt")
                batch = next(validation_set)
            state_tensor, mask, true_policy, true_value, ratings = batch

            policy_logits, value_logits = model.predict(state_tensor) 

            
            accuracy = calculate_top_k_accuracy(policy_logits+mask.log(), true_policy, k=1)
            score = round(accuracy * policy_logits.size(0))
            print(f"puzzle score: {score}, Estimated puzzle rating: {evaluate.performance_rating(ratings.flatten().tolist(), score)}")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'policy_loss': total_loss,
            }, f'checkpoints/best-pretrained-model.pt')
            
            model.train()

if __name__ == '__main__':
    # Determine the device to use: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu') 
    
    model = TakoNetConfig().create_model() # pass device to create_model for GPU
    print(f"{model.count_params():,} params")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = "checkpoints/best-pretrained-model.pt" # TODO: configure with command line args
    epoch = 0
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    print(config)
    download_training_set()
    train(model, optimizer, scheduler, starting_epoch=0)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 0
    }, f'checkpoints/best-model.pt') 