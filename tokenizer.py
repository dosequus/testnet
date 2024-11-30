# source: https://github.com/google-deepmind/searchless_chess/blob/main/src/tokenizer.py


import numpy as np
import numpy.typing as npt
import chess

# pyfmt: disable
_CHARACTERS = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'p',
    'n',
    'r',
    'k',
    'q',
    'P',
    'B',
    'N',
    'R',
    'Q',
    'K',
    'w',
    '.',
]
# pyfmt: enable
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 77

def tokenize(fen: str) -> npt.NDArray:
  """Returns an array of tokens from a fen string.

  We compute a tokenized representation of the board, from the FEN string.
  The final array of tokens is a mapping from this string to numbers, which
  are defined in the dictionary `_CHARACTERS_INDEX`.
  For the 'en passant' information, we convert the '-' (which means there is
  no en passant relevant square) to '..', to always have two characters, and
  a fixed length output.

  Args:
    fen: The board position in Forsyth-Edwards Notation.
  """
  # Extracting the relevant information from the FEN.
  board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
  board = board.replace('/', '')
  board = side + board

  indices = list()

  for char in board:
    if char in _SPACES_CHARACTERS:
      indices.extend(int(char) * [_CHARACTERS_INDEX['.']])
    else:
      indices.append(_CHARACTERS_INDEX[char])

  if castling == '-':
    indices.extend(4 * [_CHARACTERS_INDEX['.']])
  else:
    for char in castling:
      indices.append(_CHARACTERS_INDEX[char])
    # Padding castling to have exactly 4 characters.
    if len(castling) < 4:
      indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

  if en_passant == '-':
    indices.extend(2 * [_CHARACTERS_INDEX['.']])
  else:
    # En passant is a square like 'e3'.
    for char in en_passant:
      indices.append(_CHARACTERS_INDEX[char])

  # Three digits for halfmoves (since last capture) is enough since the game
  # ends at 50.
  halfmoves_last += '.' * (3 - len(halfmoves_last))
  indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

  # Three digits for full moves is enough (no game lasts longer than 999
  # moves).
  fullmoves += '.' * (3 - len(fullmoves))
  indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

  assert len(indices) == SEQUENCE_LENGTH

  return np.asarray(indices, dtype=np.uint8)

# Generate all possible UCI moves (valid or not)
def generate_uci_moves():
    files = "abcdefgh"
    promotions = "nbrq"
    moves = []
    
    board = chess.Board.empty()

    # Regular moves
    for start in range(64):
      end_squares = []
      
      board.set_piece_at(start, chess.Piece.from_symbol('Q'))
      end_squares += board.attacks(start)
      board.remove_piece_at(start)
      
      board.set_piece_at(start, chess.Piece.from_symbol('N'))
      end_squares += board.attacks(start)
      board.remove_piece_at(start)
      
      for end in end_squares:
        moves.append(chess.square_name(start) + chess.square_name(end))
                  

    # Promotion moves
    for rank, next_rank in [('2', '1'), ('7', '8')]:
      for index_file, file in enumerate(files):
        # Normal promotions.
        move = f'{file}{rank}{file}{next_rank}'
        moves.extend([(move + piece) for piece in promotions])

        # Capture promotions.
        # Left side.
        if file > 'a':
          next_file = files[index_file - 1]
          move = f'{file}{rank}{next_file}{next_rank}'
          moves.extend([(move + piece) for piece in promotions])
        # Right side.
        if file < 'h':
          next_file = files[index_file + 1]
          move = f'{file}{rank}{next_file}{next_rank}'
          moves.extend([(move + piece) for piece in promotions])

    return moves

# Create the UCI move-to-index mapping
UCI_ACTIONS = sorted(generate_uci_moves())
UCI_ACTIONS_INDEX = {move: idx for idx, move in enumerate(UCI_ACTIONS)}

# Number of unique UCI actions
NUM_ACTIONS = len(UCI_ACTIONS)

def tokenize_action(uci_move: str) -> int:
    """
    Converts a UCI move string into an integer token.

    Args:
        uci_move: A move string in UCI format (e.g., "e2e4").

    Returns:
        int: The corresponding token index.
    """
    return UCI_ACTIONS_INDEX[uci_move]