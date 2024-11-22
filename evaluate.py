import torch
import torch.backends
import torch.nn as nn
from games.chessboard import ChessGame
from model import TransformerNet
from chess import Move
import torch.optim as optim
import search
import os
import torch.functional as F
from settings import Configuration
import tqdm
from stockfish import Stockfish
from chessboard import display
import pandas as pd

config = Configuration().get_config()


def expected_score(opponent_ratings: list[float], own_rating: float) -> float:
        """How many points we expect to score in a tourney with these opponents"""
        return sum(
            1 / (1 + 10**((opponent_rating - own_rating) / 400))
            for opponent_rating in opponent_ratings
        )


def performance_rating(opponent_ratings: list[float], score: float) -> int:
    """Calculate mathematically perfect performance rating with binary search"""
    lo, hi = 0, 4000

    while hi - lo > 0.001:
        mid = (lo + hi) / 2

        if expected_score(opponent_ratings, mid) < score:
            lo = mid
        else:
            hi = mid

    return round(mid)

def selfplay_benchmark(agent1, agent2, num_games=10, device='cpu', save_path='checkpoints/best_model'):
    """
    Evaluate the new model by playing a series of games against the previously saved benchmark model.
    If the new model wins more games, it becomes the new benchmark and is saved.
    
    Args:
    - new_model (nn.Module): The new model to be evaluated.
    - num_games (int): Number of games to play between the models.
    - device (str): Device to run the models on ('cpu' or 'cuda').
    - save_path (str): Path to save the new benchmark model if it wins.
    
    Returns:
    - bool: True if the new model becomes the new benchmark, False otherwise.
    """
    game_board = display.start()

    def play_game(agent1, agent2):
        """
        Simulate a game between two models using the provided ChessGame and search2 logic.
        Returns 1 if model1 wins, 0 if it's a draw, and -1 if model2 wins.
        """
        game = ChessGame()
        if game_board: display.update(game.board.fen(), game_board)
        while not game.over():
            if game.turn:  # model1 plays as white
                _, best_move = agent1.run(game.copy(), max_depth=config.evaluation.max_depth, num_sim=5, max_nodes=config.mcts.max_nodes)
            else:  
                _, best_move = agent2.run(game.copy(), max_depth=config.evaluation.max_depth, num_sim=5, max_nodes=config.mcts.max_nodes)
            
            game.make_move(best_move)
            if game_board: display.update(game.board.fen(), game_board)
        result = game.score()
        return result  # 1 if model1 wins, 0 if draw, -1 if model2 wins

    # Load the previous model
    # previous_model = torch.load(previous_model_path, map_location=device)

    # new_model.to(device)

    agent1_wins = 0
    agent2_wins= 0
    draws = 0
    pbar = tqdm.trange(num_games)
    for game in pbar:
        # Alternate who plays as white/black
        pbar.set_description(f'+{agent1_wins}={draws}-{agent2_wins}')
        color = (-1)**game
        result = play_game(agent1, agent2)
        if result == color:
            agent1_wins += 1
        elif result == -color:
            agent2_wins += 1
        else:
            draws += 1
        if game_board: 
            display.flip(game_board)
    # if game_board: display.terminate()
    
    print(f"Agent 1 Wins: {agent1_wins}, Agent 2 Wins: {agent2_wins}, Draws: {draws}")

def stockfish_benchmark(mcts, num_games=10, device='cpu', save_path='checkpoints/best_model', game_board=None):
    """
    Evaluate the new model by playing a series of games against the previously saved benchmark model.
    If the new model wins more games, it becomes the new benchmark and is saved.
    
    Args:
    - new_model (nn.Module): The new model to be evaluated.
    - num_games (int): Number of games to play between the models.
    - device (str): Device to run the models on ('cpu' or 'cuda').
    - save_path (str): Path to save the new benchmark model if it wins.
    
    Returns:
    - bool: True if the new model becomes the new benchmark, False otherwise.
    """
    
    stockfish_rating = 100
    max_depth=config.evaluation.max_depth
    num_sim=config.evaluation.num_sim
    game_board = None if not config.visualize else display.start()

    def play_game(mcts, color):
        """
        Simulate a game between two models using the provided ChessGame and search2 logic.
        Returns 1 if model1 wins, 0 if it's a draw, and -1 if model2 wins.
        """
        game = ChessGame()
        if game_board: display.update(game.board.fen(), game_board)
        stockfish = Stockfish(depth=max_depth) # keep stockfish at the same depth
        stockfish.set_elo_rating(stockfish_rating)
        while not game.over():
            if game.turn == color:  # model1 plays as white
                _, best_move = mcts.run(game.copy(), max_depth=max_depth, num_sim=num_sim, max_nodes=config.mcts.max_nodes)
            else:  
                stockfish.set_fen_position(game.board.fen())
                best_move = Move.from_uci(stockfish.get_best_move())
            
            game.make_move(best_move)
            if game_board: display.update(game.board.fen(), game_board)
        if game_board: display.update(game.board.fen(), game_board)
        result = game.score()
        return result  # 1 if model1 wins, 0 if draw, -1 if model2 wins

    # Load the previous model
    # previous_model = torch.load(previous_model_path, map_location=device)

    # new_model.to(device)

    new_model_wins = 0
    stockfish_wins = 0
    draws = 0
    opponent_elos = []
    pbar = tqdm.trange(num_games)
    for game in pbar:
        # Alternate who plays as white/black
        pbar.set_description(f'+{new_model_wins}={draws}-{stockfish_wins}')
        color = (-1)**game
        result = play_game(mcts, color)
        opponent_elos.append(stockfish_rating)
        if result == color:
            new_model_wins += 1
        elif result == -color:
            stockfish_wins += 1
            stockfish_rating += 200
        else:
            draws += 1
        if game_board: 
            display.flip(game_board)
    # if game_board: display.terminate()
    rating = performance_rating(opponent_elos, new_model_wins+(0.5*draws))
    print(f"Tako Wins: {new_model_wins}, Stockfish Wins: {stockfish_wins}, Draws: {draws}")
    print(f"estimated elo: {rating}")
    
def puzzle_benchmark(nnet : TransformerNet, csv_path, num_puzzles = 10000):
   
    # agent = search.MCTS(nnet, explore_factor=0)
    score = 0.
    max_puzzle_win = 0
    for _, row in tqdm.tqdm(batch.iterrows(), total=num_puzzles):
        fen = row["FEN"]
        best_move_uci = row["Moves"].split()[0]  # Assume the first move is the best move
        game = ChessGame(fen)
        
        # _, pred_move = agent.run(game, num_sim=1, max_nodes=500)
        # if pred_move.uci() == best_move_uci:
        #     score += 1
        #     max_puzzle_win = max(max_puzzle_win, row['Rating'])
        state_tensor, mask_tensor = game.to_tensor(), game.get_legal_move_mask()
        policy_logits, value_logits = nnet.predict_single(state_tensor)
        
        value_probs = torch.softmax(value_logits, dim=-1)
        policy_prob = torch.softmax(policy_logits + mask_tensor)
        
        pred_moves = game.pi_to_move_map(policy_prob)
        print(pred_moves, value_probs)
        if max(pred_moves, key=pred_moves.get).uci() == best_move_uci:
            score += 1
            max_puzzle_win = max(max_puzzle_win, row['Rating'])
        # stockfish.set_fen_position(fen)
        # pred_move_uci = stockfish.get_best_move()
        # if pred_move_uci == best_move_uci:
        #     max_puzzle_win = max(max_puzzle_win, row['Rating']) 
        #     score += 1
        
        
    print("hardest puzzle solved: ", max_puzzle_win)
    print(f"puzzle score: {score}, Estimated puzzle rating: {performance_rating(batch['Rating'].to_list(), score)}")

if __name__ == '__main__':
    # Determine the device to use: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu') 
    print(device.type)
    
    selfplay_model = TransformerNet()
    finetuned_model = TransformerNet()
    
    
    selfplay_checkpoint = torch.load("checkpoints/best-model.pt")
    finetuned_checkpoint = torch.load("checkpoints/best-finetuned-model.pt")

    selfplay_model.load_state_dict(selfplay_checkpoint['model_state_dict'])
    finetuned_model.load_state_dict(finetuned_checkpoint['model_state_dict']) 
    
    print(config)
    
    puzzle_benchmark(finetuned_model, 'puzzles/lichess_db_puzzle.csv')
    # stockfish_benchmark(search.MCTS(finetuned_model, explore_factor=0), device=device.type)
    # selfplay_benchmark(search.MCTS(selfplay_model, explore_factor=0),
    #                    search.MCTS(finetuned_model, explore_factor=0))