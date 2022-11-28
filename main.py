import torch


from games.chessboard import ChessGame
from model import TransformerNet


game = ChessGame()
x = torch.from_numpy(game.state())

model = TransformerNet()
pi, v = model.predict(x, torch.ones_like(x))