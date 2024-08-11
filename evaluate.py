import torch
import torch.backends
import torch.nn as nn
from games.chessboard import ChessGame
from model import TransformerNet
from chess import Move
import torch.optim as optim
import search2
import random
from settings import load_config
import tqdm
from stockfish import Stockfish
from collections import deque

config = load_config()

def selfplay_benchmark(new_model, previous_model_path, num_games=10, device='cpu', save_path='checkpoints/best_model'):
    """
    Evaluate the new model by playing a series of games against the previously saved benchmark model.
    If the new model wins more games, it becomes the new benchmark and is saved.
    
    Returns:
    - bool: True if the new model becomes the new benchmark, False otherwise.
    """
    max_depth=config.evaluation.max_depth
    num_sim=config.evaluation.num_simulations
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
    previous_model = TransformerNet()
    checkpoint = torch.load(previous_model_path, map_location=device)
    previous_model.load_state_dict(checkpoint['model_state_dict'])
    previous_model.to(device)
    new_model.to(device)

    new_model_wins = 0
    previous_model_wins = 0
    draws = 0

    for game in range(num_games):
        break
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

def stockfish_benchmark(new_model, num_games=10, device='cpu', save_path='checkpoints/best_model'):
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
    
    STOCKFISH_RATING = 500
    max_depth=config.evaluation.max_depth
    num_sim=config.evaluation.num_simulations
    
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

    def play_game(model, color):
        """
        Simulate a game between two models using the provided ChessGame and search2 logic.
        Returns 1 if model1 wins, 0 if it's a draw, and -1 if model2 wins.
        """
        game = ChessGame()
        stockfish = Stockfish(depth=max_depth) # keep stockfish at the same depth
        stockfish.set_elo_rating(STOCKFISH_RATING)
        while not game.over():
            if game.turn == color:  # model1 plays as white
                _, best_move = search2.run(game.copy(), model, max_depth=max_depth, num_sim=num_sim)
            else:  
                stockfish.set_fen_position(game.game.fen())
                best_move = Move.from_uci(stockfish.get_best_move())
            
            game.make_move(best_move)
            print(game.game)
        
        result = game.score()
        return result  # 1 if model1 wins, 0 if draw, -1 if model2 wins

    # Load the previous model
    # previous_model = torch.load(previous_model_path, map_location=device)

    new_model.to(device)

    new_model_wins = 0
    stockfish_wins = 0
    draws = 0

    for game in tqdm.trange(num_games):
        # Alternate who plays as white/black
        if game % 2 == 0:
            result = play_game(new_model, 1)
        else:
            result = play_game(new_model, -1)
            result = -result  # Invert the result because the perspective is switched

        if result == 1:
            new_model_wins += 1
        elif result == -1:
            stockfish_wins += 1
        else:
            draws += 1

    rating = performance_rating([STOCKFISH_RATING]*num_games, new_model_wins+(0.5*draws)-stockfish_wins)
    print(f"Tako Wins: {new_model_wins}, Stockfish Wins: {stockfish_wins}, Draws: {draws}")
    print(f"estimated elo: {rating}")