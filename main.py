import torch
import cProfile, pstats
import numpy as np

from games.chessboard import ChessGame
from model import TransformerNet
import search

if __name__ == '__main__':
    test = ChessGame(fen='1b1B1k2/5B2/1q4P1/8/8/8/4K3/8')
    test.copy()
    # print(test.state, test.copy().state)
    assert np.array_equal(test.state, test.copy().state)
    # exit()
    game = ChessGame()
    nnet = TransformerNet()
    
    total_params = sum(p.numel() for p in nnet.parameters())
    print(total_params, "params")
    mcts = search.MCTS(nnet)

    with cProfile.Profile(builtins=False) as pr:
        res = mcts.run(game, num_sim=1, max_nodes=500)
        print(res)
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(20)