
import sys, os, random, time
from tqdm import tqdm

from itertools import product
import numpy as np
import IPython
import math



from games.chessboard import ChessGame
from games.gomoku import GomokuGame
from models import TrainableModel

import multiprocessing
from multiprocessing import Pool, Manager

from mcts import MCTSController
from alphazero import AlphaZeroController

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



""" CNN representing estimated value for each board state.
"""

class Net(TrainableModel):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(704, 512).float(),
            nn.ReLU(),
            nn.Linear(512, 512).float(),
            nn.ReLU(),
            nn.Linear(512, 1).float()
        )

    def loss(self, data, data_pred):
        Y_pred = data_pred["target"]
        Y_target = data["target"]
        #print ((Y_pred, Y_target))
        # return torch.tensor(0., requires_grad=True)
        return (F.mse_loss(Y_pred, Y_target))

    def forward(self, data):
        x = data['input']
        x = x.reshape(-1)
        x = self.layers(x)
        print(x.shape)
        return {'target': x}

class Net(TrainableModel):

    def __init__(self):

        super(TrainableModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 1)

    def loss(self, data, data_pred):
        Y_pred = data_pred["target"]
        Y_target = data["target"]
        #print ((Y_pred, Y_target))

        return (F.mse_loss(Y_pred, Y_target))

    def forward(self, x):
        x = x['input']
        x = x.view(-1, 2, 19, 19)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv4(x))

        x = F.max_pool2d(x, (4, 4))[:, :, 0, 0]
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.layer1(x)
        x = self.layer2(F.tanh(x))

        return {'target': x}


class TransformerNet(TrainableModel):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_hidden, max_len=64) -> None:
            super().__init__()
            pe = torch.arange(max_len).unsqueeze(1)
            self.register_buffer('pe', pe)

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:x.size(0)]
            return x
    """
        ntoken: size of tokens => 12 for chess
        emsize = 
    """
    def __init__(self, ntokens = 11, embed_size=10, d_hid=5, nlayers=2, nhead=1):
        super().__init__()
        self.transformer = nn.Transformer()
        self.pos_encoder = self.PositionalEncoding(d_hid)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(ntokens, embed_size)
        self.d_model = embed_size
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.init_weights()

    def init_weights(self) -> None:
        return
        def _init(module : nn.Module):
            initrange = 1.0
            if type(module) == nn.Linear:
                module.weight.data.uniform_(-initrange, initrange)
                if hasattr(module, 'bias'):
                    module.bias.data.zero_()
        
        _init(self.encoder)	
        self.classifier.apply(_init)
        
    def loss(self, data, data_pred):
        Y_pred = data_pred["target"]
        Y_target = data["target"]
        return (F.cross_entropy(Y_pred, Y_target))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data['input']
        """
        args:
            x, Tensor, shape [8, 8, 11]
        returns:
            outpute, Tensor Shape [1]
        """
        x = x.squeeze()
        x = x.reshape(64,1,11).squeeze()
        x = self.encoder(x)
        x = self.pos_encoder(x)
        y_pred = self.transformer_encoder(x)
        y_pred = self.classifier(y_pred)
        y_pred = y_pred.reshape(-1).sum()
        y_pred = torch.sigmoid(y_pred).unsqueeze(0)
        return {'target': y_pred}

    def checkpoint(self):
        return super().checkpoint()


if __name__ == "__main__":
    manager = Manager()
    model = TransformerNet()
    model.compile(torch.optim.Adadelta, lr=0.3)
    controller = AlphaZeroController(manager, model, T=0.2)

    for _ in range(1000):
        moves = []
        game = ChessGame("8/5k2/8/8/7Q/8/3K4/8 w - - 0 1")
        print (game)
        print ()
        try:
            while not game.over():
                move = controller.best_move(game, playouts=1)
                print(f"Moving: {move}")
                game.make_move(move)
                moves.append(move)
                print (game)
                print ()

        finally:
            print(game.game.variation_san(moves))
        
        
        

