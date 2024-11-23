from tokenizer import tokenize
import network
import torch
from games.chessboard import ChessGame
import cProfile, pstats
import search
import chess
fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"


config = network.TakoNetConfig()
nnet = config.create_model()

game = ChessGame(fen, device=nnet.device)
# checkpoint_path = "checkpoints/best-pretrained-model.pt" # TODO: configure with command line args
# checkpoint = torch.load(checkpoint_path, map_location=nnet.device)


# pred_moves = game.create_move_map(pprobs)
mcts = search.MCTS(nnet)

with cProfile.Profile(builtins=False) as pr:
    res = mcts.run(search.Node(None, fen, 0), num_sim=100, think_time=.1)

    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)