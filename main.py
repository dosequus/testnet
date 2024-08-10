import torch


from games.chessboard import ChessGame
from model import TransformerNet


game = ChessGame("6k1/5p2/6p1/8/7p/8/6PP/6K1 b - - 0 0")
x = torch.from_numpy(game.state)
moves = game.valid_moves()
model = TransformerNet()
pi, v = model.predict(x, TransformerNet.get_legal_move_mask(moves, game.game.piece_map()))
pi_star = TransformerNet.pi_to_move_map(pi, moves, game.game.piece_map())
