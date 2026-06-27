"""
Play chess against your trained AlphaZero AI in the terminal.

Run:  python play.py
"""

import chess
import torch
from environment import encode_board
from move_encoder import move_to_index
from neural_net import AlphaZeroNet
from mcts import MCTS
from config import MCTS_SIMULATIONS_PLAY


def print_board(board: chess.Board):
    """Pretty-print the board from White's perspective."""
    print("\n  a b c d e f g h")
    print(" +----------------+")
    for rank in range(7, -1, -1):
        row = f"{rank + 1}|"
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                row += " ."
            else:
                symbol = piece.symbol()
                row += f" {symbol}"
        row += f" |{rank + 1}"
        print(row)
    print(" +----------------+")
    print("  a b c d e f g h\n")


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = AlphaZeroNet().to(device)
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=device, weights_only=True))
        print("✓ Loaded trained model.")
    except FileNotFoundError:
        print("⚠ No trained model found. Using random weights.")
        print("  Run  python train.py  first for a real opponent.\n")

    model.eval()
    mcts = MCTS(model, device)

    board = chess.Board()

    print("=" * 40)
    print("  AlphaZero Chess — Terminal Game")
    print("  You play as White (uppercase pieces)")
    print("  Type moves in UCI format, e.g. e2e4")
    print("  Type 'quit' to exit")
    print("=" * 40)

    while not board.is_game_over():
        print_board(board)

        if board.turn == chess.WHITE:
            # ── Human turn ────────────────────────────────
            legal = [m.uci() for m in board.legal_moves]
            while True:
                user_input = input("Your move: ").strip().lower()
                if user_input == "quit":
                    print("Goodbye!")
                    return
                if user_input in legal:
                    break
                print(f"Invalid. Legal moves: {', '.join(sorted(legal)[:10])}...")

            board.push_uci(user_input)

        else:
            # ── AI turn ───────────────────────────────────
            print("AI is thinking...")
            action_probs = mcts.search(board, MCTS_SIMULATIONS_PLAY, add_noise=False)
            move = mcts.pick_move(board, action_probs, temperature=0.0)
            print(f"AI plays: {move.uci()}")
            board.push(move)

    # ── Game Over ──────────────────────────────────────────
    print_board(board)
    result = board.result()
    if result == "1-0":
        print("White wins!")
    elif result == "0-1":
        print("Black wins!")
    else:
        print("Draw!")


if __name__ == "__main__":
    play()