import torch


from games.chessboard import ChessGame
from model import TransformerNet


game = ChessGame()
x = torch.from_numpy(game.state())
moves = game.valid_moves()

model = TransformerNet()
pi, v = model.predict(x, TransformerNet.get_legal_move_mask(moves, game.game.piece_map()))
print(pi, v)
pi_star = TransformerNet.pi_to_move_map(pi, moves, game.game.piece_map())
print(sum(pi_star.values()))