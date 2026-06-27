"""
AlphaZero move encoding / decoding.

Action space: 4 672 = 64 from-squares × 73 move types
  Move types 0-55 : Queen-like  (8 directions × 7 distances)
  Move types 56-63: Knight      (8 offsets)
  Move types 64-72: Underpromo  (3 pieces × 3 capture-directions)
"""

import chess
import numpy as np

# 8 queen directions: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [
    ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1),
    (-1,  0), (-1, -1), ( 0, -1), ( 1, -1),
]

# 8 knight offsets
KNIGHT_OFFSETS = [
    (-2, -1), (-2,  1), (-1, -2), (-1,  2),
    ( 1, -2), ( 1,  2), ( 2, -1), ( 2,  1),
]

# Underpromotion piece types (queen promotion is covered by queen moves)
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

ACTION_SIZE = 4672


def move_to_index(move: chess.Move) -> int:
    """
    Encode a python-chess Move into an integer in [0, 4672).

    Args:
        move: A legal chess.Move.

    Returns:
        Integer action index.
    """
    from_sq = move.from_square
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)
    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)
    dr = to_rank - from_rank
    dc = to_file - from_file

    # ── Underpromotion (not queen) ──────────────────────────
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        dir_idx = dc + 1                       # -1→0, 0→1, +1→2
        move_type = 64 + piece_idx * 3 + dir_idx
        return from_sq * 73 + move_type

    # ── Knight move ────────────────────────────────────────
    if (dr, dc) in KNIGHT_OFFSETS:
        move_type = 56 + KNIGHT_OFFSETS.index((dr, dc))
        return from_sq * 73 + move_type

    # ── Queen move (includes queen promotion & normal moves) ─
    if dr == 0:
        direction = (0, 1 if dc > 0 else -1)
        distance = abs(dc)
    elif dc == 0:
        direction = (1 if dr > 0 else -1, 0)
        distance = abs(dr)
    else:
        direction = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
        distance = abs(dr)

    dir_idx = QUEEN_DIRECTIONS.index(direction)
    move_type = dir_idx * 7 + (distance - 1)
    return from_sq * 73 + move_type


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """
    Decode an integer index in [0, 4672) back to a python-chess Move.

    Returns None if the decoded move would land off the board.
    """
    from_sq = index // 73
    move_type = index % 73
    from_rank = chess.square_rank(from_sq)
    from_file = chess.square_file(from_sq)

    # ── Queen move ─────────────────────────────────────────
    if move_type < 56:
        dir_idx = move_type // 7
        distance = (move_type % 7) + 1
        dr, dc = QUEEN_DIRECTIONS[dir_idx]
        to_rank = from_rank + dr * distance
        to_file = from_file + dc * distance
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            return None
        to_sq = chess.square(to_file, to_rank)
        # Queen promotion for advancing pawns
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return chess.Move(from_sq, to_sq)

    # ── Knight move ────────────────────────────────────────
    if move_type < 64:
        dr, dc = KNIGHT_OFFSETS[move_type - 56]
        to_rank = from_rank + dr
        to_file = from_file + dc
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            return None
        to_sq = chess.square(to_file, to_rank)
        return chess.Move(from_sq, to_sq)

    # ── Underpromotion ─────────────────────────────────────
    up_idx = move_type - 64
    piece_idx = up_idx // 3
    dir_idx = up_idx % 3
    promo_piece = UNDERPROMO_PIECES[piece_idx]
    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.PAWN:
        return None
    forward = 1 if piece.color == chess.WHITE else -1
    to_rank = from_rank + forward
    to_file = from_file + (dir_idx - 1)
    if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
        return None
    to_sq = chess.square(to_file, to_rank)
    return chess.Move(from_sq, to_sq, promotion=promo_piece)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean mask of shape (4672,) — True for legal moves."""
    mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask