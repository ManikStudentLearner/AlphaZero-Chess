"""
Chess environment wrapper using python-chess.
Encodes the board into an 18-channel tensor from the current player's
perspective (canonical form) so the neural network always sees
"my pieces" in channels 0-5 regardless of color.

Channel layout:
  0-5   : Current player's P, N, B, R, Q, K
  6-11  : Opponent's      P, N, B, R, Q, K
  12    : Current player's kingside castling right
  13    : Current player's queenside castling right
  14    : Opponent's kingside castling right
  15    : Opponent's queenside castling right
  16    : En passant target square
  17    : Side-to-move indicator (1 = white, 0 = black)
"""

import chess
import numpy as np


PIECE_INDEX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Convert a python-chess Board into an (18, 8, 8) float32 array
    in canonical form (current player = channels 0-5).

    Args:
        board: A chess.Board instance.

    Returns:
        np.ndarray of shape (18, 8, 8) with values in {0.0, 1.0}.
    """
    state = np.zeros((18, 8, 8), dtype=np.float32)

    current = board.turn        # chess.WHITE or chess.BLACK
    opponent = not current

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)   # 0-7
            file = chess.square_file(square)   # 0-7
            offset = 0 if piece.color == current else 6
            channel = offset + PIECE_INDEX[piece.piece_type]
            state[channel][rank][file] = 1.0

    # Castling rights — current player first, then opponent
    if board.has_kingside_castling_rights(current):
        state[12] = 1.0
    if board.has_queenside_castling_rights(current):
        state[13] = 1.0
    if board.has_kingside_castling_rights(opponent):
        state[14] = 1.0
    if board.has_queenside_castling_rights(opponent):
        state[15] = 1.0

    # En passant target square
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        state[16][ep_rank][ep_file] = 1.0

    # Side-to-move indicator
    if current == chess.WHITE:
        state[17] = 1.0

    return state


def get_game_value(board: chess.Board) -> float:
    """
    Return the terminal value from the CURRENT PLAYER's perspective.

    Returns:
        +1.0 if the current player wins,
        -1.0 if the current player loses,
         0.0 for a draw.
    """
    result = board.result()
    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    elif result == "0-1":
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0