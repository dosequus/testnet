# source: https://github.com/google-deepmind/searchless_chess/blob/main/src/tokenizer.py


import numpy as np
import numpy.typing as npt

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
    ranks = "12345678"
    promotions = "nbrq"
    moves = []

    # Regular moves
    for start_file in files:
        for start_rank in ranks:
            for end_file in files:
                for end_rank in ranks:
                    moves.append(f"{start_file}{start_rank}{end_file}{end_rank}")

    # Promotion moves
    for start_file in files:
        for end_file in files:
            for promotion in promotions:
                # White pawn promotions
                moves.append(f"{start_file}7{end_file}8{promotion}")
                # Black pawn promotions
                moves.append(f"{start_file}2{end_file}1{promotion}")

    # Castling moves (standard)
    moves.append("e1g1")  # White kingside
    moves.append("e1c1")  # White queenside
    moves.append("e8g8")  # Black kingside
    moves.append("e8c8")  # Black queenside

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