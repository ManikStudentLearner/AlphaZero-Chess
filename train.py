"""
AlphaZero self-play training loop WITH live GUI observer, Human Game importer,
Continuous self-improvement, AND Persistent Replay Buffer.
"""

import os
import chess
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import tkinter as tk
import time
import sys
import pickle  # NEW: For saving the replay buffer

from environment import encode_board, get_game_value
from move_encoder import move_to_index, legal_move_mask, ACTION_SIZE
from neural_net import AlphaZeroNet
from mcts import MCTS
from config import *

# ─── File Paths ───────────────────────────────────────────
BUFFER_FILE = "models/replay_buffer.pkl"

# ─── GUI Constants ────────────────────────────────────────
SQ_SIZE = 80
BOARD_PIXELS = SQ_SIZE * 8
LIGHT_SQ_COLOR = "#F0D9B5"
DARK_SQ_COLOR = "#B58863"
BG_COLOR = "#302E2B"

PIECE_UNICODE = {
    chess.PAWN:   {'white': '♙', 'black': '♟'},
    chess.KNIGHT: {'white': '♘', 'black': '♞'},
    chess.BISHOP: {'white': '♗', 'black': '♝'},
    chess.ROOK:   {'white': '♖', 'black': '♜'},
    chess.QUEEN:  {'white': '♕', 'black': '♛'},
    chess.KING:   {'white': '♔', 'black': '♚'},
}


class TrainObserverGUI:
    """A read-only Tkinter GUI that watches the AI play against itself."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AlphaZero Self-Play Training (Observing)")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.is_running = True
        
        self.canvas = tk.Canvas(self.root, width=BOARD_PIXELS, height=BOARD_PIXELS + 40, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack()
        
        self.board = chess.Board()
        self.message = "Initializing..."
        self.draw_board()

    def on_close(self):
        self.is_running = False
        self.root.destroy()

    def update_board(self, board: chess.Board, message: str = ""):
        if not self.is_running:
            return
            
        self.board = board
        self.message = message
        self.draw_board()
        
        try:
            self.root.update()
        except tk.TclError:
            self.is_running = False

    def draw_board(self):
        self.canvas.delete("all")
        
        for rank in range(8):
            for file in range(8):
                x1 = file * SQ_SIZE
                y1 = (7 - rank) * SQ_SIZE
                x2 = x1 + SQ_SIZE
                y2 = y1 + SQ_SIZE
                color = LIGHT_SQ_COLOR if (rank + file) % 2 == 0 else DARK_SQ_COLOR
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                cx = file * SQ_SIZE + SQ_SIZE // 2
                cy = (7 - rank) * SQ_SIZE + SQ_SIZE // 2
                symbol = PIECE_UNICODE[piece.piece_type]['white' if piece.color == chess.WHITE else 'black']
                
                if piece.color == chess.WHITE:
                    self.canvas.create_text(cx+1, cy+1, text=symbol, font=("Arial", 50), fill="black")
                    self.canvas.create_text(cx, cy, text=symbol, font=("Arial", 50), fill="white")
                else:
                    self.canvas.create_text(cx+1, cy+1, text=symbol, font=("Arial", 50), fill="gray")
                    self.canvas.create_text(cx, cy, text=symbol, font=("Arial", 50), fill="black")

        self.canvas.create_rectangle(0, BOARD_PIXELS, BOARD_PIXELS, BOARD_PIXELS + 40, fill=BG_COLOR, outline="")
        self.canvas.create_text(15, BOARD_PIXELS + 20, text=self.message, anchor="w", fill="white", font=("Arial", 14))


# ─── Training Logic ────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        states, policies, values = zip(*(self.buffer[i] for i in indices))
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )

    def save_to_disk(self, filepath: str):
        """Saves the replay buffer to a file so it persists between runs."""
        print(f"  💾 Saving replay buffer ({len(self.buffer)} positions) to disk...")
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load_from_disk(self, filepath: str):
        """Loads the replay buffer from a file."""
        if os.path.exists(filepath):
            print(f"  💾 Loading previous replay buffer from disk...")
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
                for item in saved_data:
                    self.add(*item)
            print(f"  ✓ Loaded {len(self.buffer)} past training positions.")


def import_human_games(buffer: ReplayBuffer) -> int:
    if not os.path.exists("human_data"):
        return 0
        
    files = [f for f in os.listdir("human_data") if f.endswith(".npz")]
    if not files:
        return 0
        
    count = 0
    for f in files:
        filepath = os.path.join("human_data", f)
        data = np.load(filepath)
        states = data['states']
        policies = data['policies']
        values = data['values']
        
        for i in range(len(states)):
            buffer.add(states[i], policies[i], values[i])
            count += 1
            
        os.remove(filepath)
        
    print(f"  ✓ Imported {count} positions from {len(files)} human games.")
    return count


def self_play_game(model: AlphaZeroNet, mcts: MCTS, game_num: int, gui: TrainObserverGUI) -> list[tuple]:
    board = chess.Board()
    examples = []
    move_count = 0

    print(f"  Playing game {game_num}: ", end="", flush=True)
    
    while not board.is_game_over() and move_count < MAX_MOVES:
        temp = TEMPERATURE if move_count < TEMPERATURE_THRESHOLD else 0.0
        action_probs = mcts.search(board, MCTS_SIMULATIONS, add_noise=True)
        move = mcts.pick_move(board, action_probs, temperature=temp)
        
        print(f"{move.uci()} ", end="", flush=True)

        state = encode_board(board)
        examples.append((state, action_probs, board.turn))

        board.push(move)
        move_count += 1
        
        turn_name = "White" if board.turn == chess.WHITE else "Black"
        gui_msg = f"Game {game_num} | Move {move_count} | {turn_name} to play"
        gui.update_board(board, gui_msg)
        
        time.sleep(0.1) 
        
        if not gui.is_running:
            print("\n[GUI closed. Stopping training script.]")
            sys.exit(0)
        
    print(f" | Result: {board.result()}")

    final_value_white = get_game_value(board)
    if board.turn == chess.WHITE:
        value_from_white = final_value_white   
    else:
        value_from_white = -final_value_white  

    training_data = []
    for state, policy, player in examples:
        value = value_from_white if player == chess.WHITE else -value_from_white
        training_data.append((state, policy, value))

    return training_data


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = AlphaZeroNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # ── NEW: Initialize Buffer and Load Past Data ──────────
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    buffer.load_from_disk(BUFFER_FILE) # Loads all the old AI vs AI games!

    os.makedirs("models", exist_ok=True)
    os.makedirs("human_data", exist_ok=True)

    # ── RESUME TRAINING FROM LAST CHECKPOINT ─────────────
    start_iteration = 1
    checkpoint_path = "models/best_model.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Found existing trained model at {checkpoint_path}.")
        print("  Resuming training! The AI is getting smarter...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint.get("iteration", 0) + 1
        print(f"  Starting at overall iteration {start_iteration}.\n")
    else:
        print("\n⚠ No existing model found. Starting training from scratch (random AI).\n")
    # ────────────────────────────────────────────────────────

    # ── Launch the Observer GUI ─────────────────────────────
    gui = TrainObserverGUI()
    gui.update_board(chess.Board(), "Starting self-play...")
    time.sleep(1)

    try: # Wrap in try/except so we save the buffer even if the user Ctrl+C's the script!
        for i in range(1, NUM_ITERATIONS + 1):
            iteration = start_iteration + i - 1 
            
            print(f"\n{'='*50}")
            print(f"ITERATION {i}/{NUM_ITERATIONS} (Overall Iteration: {iteration})")
            print(f"Current Brain Memory: {len(buffer)} positions")
            print(f"{'='*50}")

            # ── Self-Play Phase ────────────────────────────────
            model.eval()
            mcts = MCTS(model, device)

            # ── Load human games first! ────────────────────────
            import_human_games(buffer)

            for game_num in range(1, GAMES_PER_ITERATION + 1):
                game_data = self_play_game(model, mcts, game_num, gui)
                for state, policy, value in game_data:
                    buffer.add(state, policy, value)
                    
                if not gui.is_running:
                    raise KeyboardInterrupt # Break out to save buffer

            # ── Training Phase ─────────────────────────────────
            model.train()
            total_loss = 0.0
            
            if gui.is_running:
                gui.update_board(chess.Board(), "Training Neural Network...")

            for epoch in range(1, EPOCHS_PER_ITERATION + 1):
                states, policies, values = buffer.sample(BATCH_SIZE)

                s_tensor = torch.tensor(states).to(device)
                p_tensor = torch.tensor(policies).to(device)
                v_tensor = torch.tensor(values).unsqueeze(1).to(device)

                pred_logits, pred_value = model(s_tensor)

                policy_loss = F.cross_entropy(pred_logits, p_tensor)
                value_loss = F.mse_loss(pred_value, v_tensor)

                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(f"    Epoch {epoch}/{EPOCHS_PER_ITERATION}  "
                      f"Loss: {loss.item():.4f}  "
                      f"(P: {policy_loss.item():.4f}  V: {value_loss.item():.4f})")

            avg_loss = total_loss / EPOCHS_PER_ITERATION

            # ── Checkpoint ─────────────────────────────────────
            save_path = f"models/iteration_{iteration:03d}.pth"
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
            }, save_path)

            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
            }, "models/best_model.pth")
            
            print(f"  ✓ Saved checkpoint → {save_path}")
            
            # ── NEW: Save Buffer after every iteration! ──────
            buffer.save_to_disk(BUFFER_FILE)

    except KeyboardInterrupt:
        print("\n\n[Training Interrupted by User. Saving progress...]")
        buffer.save_to_disk(BUFFER_FILE)
        print("Progress saved. You can safely resume later!")
        sys.exit(0)


if __name__ == "__main__":
    train()