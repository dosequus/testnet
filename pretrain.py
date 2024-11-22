import torch
import torch.backends
from games.chessboard import ChessGame
from model import TransformerNet
import torch.optim as optim
from settings import Configuration
import tqdm
import pandas as pd
import os
import zstandard as zstd
import requests
from concurrent.futures import ProcessPoolExecutor
import evaluate
from chess import Move
from network import TakoNet, TakoNetConfig
from tokenizer import tokenize

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
        best_move = row["Moves"].split()[0]
        game = ChessGame(fen)

        # State tensor
        state_tensor = game.to_tensor()

        # Policy tensor
        pi = {Move.from_uci(best_move): torch.tensor(1.0)}
        policy_tensor = game.create_sparse_policy(pi)

        # Value tensor placeholder (replace with actual WDL if available)
        turn = game.turn
        wdl = [0.8, 0.1, 0.1] if turn == 1 else [0.1, 0.1, 0.8]  # Example values
        value_tensor = torch.tensor(wdl, dtype=torch.float)

        return state_tensor, policy_tensor, value_tensor

def preprocess_data(puzzles: pd.DataFrame, save_path: str = "puzzles/preprocessed_chess_puzzles.pt"):
    """
    Preprocess the entire chess puzzles dataset into tensors and save to disk.
    Args:
        puzzles (pd.DataFrame): DataFrame with columns 'FEN' and 'Moves'.
        save_path (str): Path to save the preprocessed tensors.
    Returns:
        None
    """
    # Preprocess the dataset in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(preprocess_row, [row for _, row in puzzles.iterrows()]))

    # Unpack results and save tensors
    state_tensors, policy_tensors, value_tensors = zip(*results)
    torch.save({
        "state_tensors": torch.stack(state_tensors),
        "policy_tensors": torch.stack(policy_tensors),
        "value_tensors": torch.stack(value_tensors),
    }, save_path)

    print(f"Preprocessed data saved to {save_path}")


def compile_batch(batch_size: int, preprocessed_path: str = "puzzles/preprocessed_chess_puzzles.pt", device = 'cpu'):
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

    # Sample random indices for the batch
    indices = torch.randperm(len(state_tensors))[:batch_size]

    # Return batched tensors
    return (
        state_tensors[indices],
        policy_tensors[indices],
        value_tensors[indices],
    )


def train(model: TakoNet, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, starting_epoch=0):    
    PUZZLE_THRESHOLD = 1500
    csv_path = "./puzzles/lichess_db_puzzle.csv"
    print("Loading puzzles to memory...")
    puzzle_dataset = pd.read_csv(csv_path, nrows=100_000, usecols=['FEN', 'Moves', 'Rating'])
    puzzle_dataset = puzzle_dataset[puzzle_dataset['Rating'] < PUZZLE_THRESHOLD]
    NUM_PUZZLES = len(puzzle_dataset)
    
    print(f"Preprocessing {NUM_PUZZLES:,} rows...")
    preprocess_data(puzzle_dataset)
    print("Starting pre-training on...")
    pbar = tqdm.tqdm(total=config.pretrain.num_epochs)
    for epoch in range(starting_epoch, config.pretrain.num_epochs):           
        batch = compile_batch(config.pretrain.batch_size, device=model.device)
        if batch:
            state_tensor, true_policy, true_value = batch
            # Forward pass
            policy_logits, value_logits = model(state_tensor)
            # Compute loss
            # value_loss = nn.CrossEntropyLoss()(value_logits, true_value)
            policy_loss = model.policy_loss(policy_logits, true_policy)
            total_loss = policy_loss
            # with open("logs/pretraining/value.csv", 'a+') as logfile:
            #     logfile.write(f"{epoch},{value_loss.item()}\n")
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.update()
        
        with open("logs/pretraining/policy.csv", 'a+') as logfile:
                    logfile.write(f"{epoch},{policy_loss.item()}\n")
        scheduler.step()
        pbar.set_description_str(f"loss={total_loss.item():.3f}, lr={scheduler.get_last_lr()[0]:.4f}")
        
        if (epoch+1) % config.pretrain.evaluation_interval == 0:
            model.eval()
            puzzle_batch = puzzle_dataset.sample(config.pretrain.validation_batch_size)
            score = 0
            max_puzzle_win = 0
            for _, row in puzzle_batch.iterrows():
                fen = row["FEN"]
                best_move_uci = row["Moves"].split()[0]  # Assume the first move is the best move
                game = ChessGame(fen, model.device)             
                policy_logits, value_logits = model.predict_single(game.to_tensor())
                
                # value_probs = value_logits.softmax(-1)
                policy_probs = (policy_logits + game.get_legal_move_mask()).softmax(-1)

                pred_moves = game.create_move_map(policy_probs)
                if max(pred_moves, key=pred_moves.get).uci() == best_move_uci:
                    score += 1
                    max_puzzle_win = max(max_puzzle_win, row['Rating'])
            print("hardest puzzle solved: ", max_puzzle_win)
            print(f"puzzle score: {score}, Estimated puzzle rating: {evaluate.performance_rating(puzzle_batch['Rating'].to_list(), score)}")
            
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
    optimizer = optim.Adam(model.parameters(), lr=0.2)
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
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    print(config)
    download_training_set()
    train(model, optimizer, scheduler, starting_epoch=0)