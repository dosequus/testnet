import torch
import cProfile, pstats
import numpy as np

from games.chessboard import ChessGame
from model import TransformerNet
import search
import multiprocessing

if __name__ == '__main__':
    test = ChessGame(fen=None)
    test.copy()
    # print(test.state, test.copy().state)
    assert np.array_equal(test.state, test.copy().state)
    # exit()
    game = ChessGame()
    nnet = TransformerNet()
    mcts = search.MCTS(nnet)

    with cProfile.Profile(builtins=False) as pr:
        mcts.run(game, num_sim=1, max_nodes=80)
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)