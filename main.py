
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




class TransformerNet(TrainableModel):
	class PositionalEncoding(nn.Module):
		def __init__(self, d_hidden, max_len=64) -> None:
			super().__init__()
			position = torch.arange(max_len).unsqueeze(1)
			div_term = torch.exp(torch.arange(0, d_hidden, 2) * (-math.log(10000.0) / d_hidden))
			pe = torch.zeros(max_len, 1, d_hidden)
			pe[:, 0, 0::2] = torch.sin(position * div_term)
			pe[:, 0, 1::2] = torch.cos(position * div_term)
			self.register_buffer('pe', pe)

		def forward(self, x : torch.Tensor) -> torch.Tensor:
			x = x + self.pe[:x.size(0)]
			return x
	"""
		ntoken: size of tokens => 12 for chess
		emsize = 
	"""
	def __init__(self, ntokens = 11, embed_size=50, d_hid=50, nlayers=2, nhead=2):
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
		def _init(module : nn.Module):
			initrange = 0.1
			if type(module) == nn.Linear:
				module.weight.data.uniform_(-initrange, initrange)
				if hasattr(module, 'bias'):
					module.bias.data.zero_()
		
		_init(self.encoder)	
		self.classifier.apply(_init)
		
	def loss(self, data, data_pred):
		Y_pred = data_pred["target"]
		Y_target = data["target"]
		return (F.mse_loss(Y_pred, Y_target))

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
		y_pred = F.gelu(y_pred).unsqueeze(0)
		return {'target': y_pred}


if __name__ == "__main__":
	manager = Manager()
	model = TransformerNet()
	model.compile(torch.optim.Adadelta, lr=0.3)
	controller = AlphaZeroController(manager, model, T=0.2)

	for i in tqdm(range(0, 1000)):
		game = ChessGame()
		game.make_move(random.choice(list(game.valid_moves())))
		print (game)
		print ()

		while not game.over():
			game.make_move(controller.best_move(game, playouts=1))

			print (game)
			print ()

