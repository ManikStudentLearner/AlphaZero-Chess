"""
Graphical User Interface to play against your AlphaZero AI.
Now supports playing as White OR Black! Saves games for AI training.
"""

import chess
import torch
import tkinter as tk
import numpy as np
import os
import time
from environment import encode_board, get_game_value
from move_encoder import move_to_index, ACTION_SIZE
from neural_net import AlphaZeroNet
from mcts import MCTS
from config import MCTS_SIMULATIONS_PLAY

# ─── UI Constants ────────────────────────────────────────
SQ_SIZE = 80
BOARD_PIXELS = SQ_SIZE * 8

LIGHT_SQ_COLOR = "#F0D9B5"
DARK_SQ_COLOR = "#B58863"
HIGHLIGHT_COLOR = "#F6F669"
GREEN_DOT = "#00CC00"
BG_COLOR = "#302E2B"

PIECE_UNICODE = {
    chess.PAWN:   {'white': '♙', 'black': '♟'},
    chess.KNIGHT: {'white': '♘', 'black': '♞'},
    chess.BISHOP: {'white': '♗', 'black': '♝'},
    chess.ROOK:   {'white': '♖', 'black': '♜'},
    chess.QUEEN:  {'white': '♕', 'black': '♛'},
    chess.KING:   {'white': '♔', 'black': '♚'},
}


def choose_color():
    """Opens a pop-up window to let the user pick their color."""
    choice = None
    
    def set_color(c):
        nonlocal choice
        choice = c
        root.destroy()
        
    root = tk.Tk()
    root.title("Choose Your Color")
    root.resizable(False, False)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    tk.Label(root, text="Play as:", font=("Arial", 16)).pack(pady=10)
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    
    tk.Button(btn_frame, text="⬜ White", font=("Arial", 14), width=10, 
              command=lambda: set_color(chess.WHITE)).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="⬛ Black", font=("Arial", 14), width=10, 
              command=lambda: set_color(chess.BLACK)).pack(side=tk.RIGHT, padx=10)
              
    root.mainloop()
    
    # Default to White if they close the window without choosing
    return choice if choice is not None else chess.WHITE


class ChessGUI:
    def __init__(self, root, human_color: chess.Color):
        self.root = root
        self.root.title("AlphaZero Chess AI")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=BOARD_PIXELS, height=BOARD_PIXELS + 40, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack()

        # ─── Player Colors ────────────────────────────────
        self.human_color = human_color
        self.ai_color = not human_color
        self.flipped = (self.human_color == chess.BLACK) # Flip board if playing Black
        
        color_name = "White" if self.human_color == chess.WHITE else "Black"
        self.root.title(f"AlphaZero Chess AI (You are {color_name})")

        # Load AI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet().to(self.device)
        try:
            self.model.load_state_dict(torch.load("models/best_model.pth", map_location=self.device, weights_only=True))
            self.ai_status = "AI: Trained Model Loaded"
            print("\n✓ Loaded trained model.")
        except FileNotFoundError:
            self.ai_status = "AI: No model found (Playing random)"
            print("\n⚠ No trained model found. Playing against random weights.")
        
        self.model.eval()
        self.mcts = MCTS(self.model, self.device)

        # Game State
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.message = "Your turn" if self.human_color == chess.WHITE else "AI is thinking..."
        
        self.game_examples = []

        self.canvas.bind("<Button-1>", self.handle_click)

        print(f"\n--- Game Started (You are {color_name}) ---")
        self.draw_board()

        # If human is Black, make AI play the first move immediately
        if self.human_color == chess.BLACK:
            self.root.update()
            self.make_ai_move()

    def draw_board(self):
        self.canvas.delete("all")
        
        # LAYER 1: Draw Squares
        for rank in range(8):
            for file in range(8):
                # Calculate drawing coordinates based on board flip
                if self.flipped:
                    x1 = (7 - file) * SQ_SIZE
                    y1 = rank * SQ_SIZE
                else:
                    x1 = file * SQ_SIZE
                    y1 = (7 - rank) * SQ_SIZE
                    
                x2 = x1 + SQ_SIZE
                y2 = y1 + SQ_SIZE
                color = LIGHT_SQ_COLOR if (rank + file) % 2 == 0 else DARK_SQ_COLOR
                
                # Highlight selected square
                if self.selected_square == chess.square(file, rank):
                    color = HIGHLIGHT_COLOR
                    
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # LAYER 2: Draw Legal Move Indicators
        for move in self.legal_moves_for_selected:
            target_file = chess.square_file(move.to_square)
            target_rank = chess.square_rank(move.to_square)
            
            if self.flipped:
                cx = (7 - target_file) * SQ_SIZE + SQ_SIZE // 2
                cy = target_rank * SQ_SIZE + SQ_SIZE // 2
            else:
                cx = target_file * SQ_SIZE + SQ_SIZE // 2
                cy = (7 - target_rank) * SQ_SIZE + SQ_SIZE // 2
                
            if self.board.is_capture(move):
                self.canvas.create_oval(cx - 35, cy - 35, cx + 35, cy + 35, outline=GREEN_DOT, width=6)
            else:
                self.canvas.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, fill=GREEN_DOT, outline=GREEN_DOT)

        # LAYER 3: Draw Pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                
                if self.flipped:
                    cx = (7 - file) * SQ_SIZE + SQ_SIZE // 2
                    cy = rank * SQ_SIZE + SQ_SIZE // 2
                else:
                    cx = file * SQ_SIZE + SQ_SIZE // 2
                    cy = (7 - rank) * SQ_SIZE + SQ_SIZE // 2
                    
                symbol = PIECE_UNICODE[piece.piece_type]['white' if piece.color == chess.WHITE else 'black']
                
                if piece.color == chess.WHITE:
                    self.canvas.create_text(cx+1, cy+1, text=symbol, font=("Arial", 50), fill="black")
                    self.canvas.create_text(cx, cy, text=symbol, font=("Arial", 50), fill="white")
                else:
                    self.canvas.create_text(cx+1, cy+1, text=symbol, font=("Arial", 50), fill="gray")
                    self.canvas.create_text(cx, cy, text=symbol, font=("Arial", 50), fill="black")

        # LAYER 4: Status Bar
        self.canvas.create_rectangle(0, BOARD_PIXELS, BOARD_PIXELS, BOARD_PIXELS + 40, fill=BG_COLOR, outline="")
        self.canvas.create_text(15, BOARD_PIXELS + 20, text=f"{self.message}  |  {self.ai_status}", anchor="w", fill="white", font=("Arial", 14))

    def handle_click(self, event):
        # Ignore clicks if it's the AI's turn or game is over
        if self.board.turn != self.human_color or self.board.is_game_over():
            return

        x, y = event.x, event.y
        if y >= BOARD_PIXELS: return

        # Convert screen coordinates to board file/rank
        if self.flipped:
            file = 7 - (x // SQ_SIZE)
            rank = y // SQ_SIZE
        else:
            file = x // SQ_SIZE
            rank = 7 - (y // SQ_SIZE)
            
        clicked_square = chess.square(file, rank)

        if self.selected_square is not None:
            move_found = None
            for move in self.legal_moves_for_selected:
                if move.to_square == clicked_square:
                    if move.promotion is not None:
                        move_found = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)
                    else:
                        move_found = move
                    break

            if move_found and move_found in self.board.legal_moves:
                # ─── Record Human Move ──────────────────
                state = encode_board(self.board)
                player = self.board.turn
                policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                policy[move_to_index(move_found)] = 1.0
                self.game_examples.append((state, policy, player))

                self.board.push(move_found)
                color_name = "White" if self.human_color == chess.WHITE else "Black"
                print(f"Human ({color_name}) plays: {move_found.uci()}")
                
                self.selected_square = None
                self.legal_moves_for_selected = []
                
                if not self.board.is_game_over():
                    self.message = "AI is thinking..."
                    self.draw_board()
                    self.root.update()  
                    self.make_ai_move()
                else:
                    self.end_game()
            else:
                # Clicked invalid target, select new piece or deselect
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.human_color:
                    self.selected_square = clicked_square
                    self.legal_moves_for_selected = [m for m in self.board.legal_moves if m.from_square == clicked_square]
                else:
                    self.selected_square = None
                    self.legal_moves_for_selected = []
                self.draw_board()
        else:
            # Select a piece
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.human_color:
                self.selected_square = clicked_square
                self.legal_moves_for_selected = [m for m in self.board.legal_moves if m.from_square == clicked_square]
                self.draw_board()

    def make_ai_move(self):
        state = encode_board(self.board)
        player = self.board.turn
        
        action_probs = self.mcts.search(self.board, MCTS_SIMULATIONS_PLAY, add_noise=False)
        move = self.mcts.pick_move(self.board, action_probs, temperature=0.0)
        
        # ─── Record AI Move ──────────────────────
        self.game_examples.append((state, action_probs, player))

        self.board.push(move)
        color_name = "White" if self.ai_color == chess.WHITE else "Black"
        print(f"AI ({color_name}) plays:     {move.uci()}")
        print("-" * 20)
        
        if self.board.is_game_over():
            self.end_game()
        else:
            self.message = "Your turn"
            self.draw_board()

    def end_game(self):
        result = self.board.result()
        
        # ─── Save the game for training ─────────
        final_value_white = get_game_value(self.board)
        if self.board.turn == chess.WHITE:
            value_from_white = final_value_white   
        else:
            value_from_white = -final_value_white  

        training_data = []
        for state, policy, player in self.game_examples:
            value = value_from_white if player == chess.WHITE else -value_from_white
            training_data.append((state, policy, value))

        if training_data:
            os.makedirs("human_data", exist_ok=True)
            filename = f"human_data/game_{int(time.time())}.npz"
            states = np.array([d[0] for d in training_data])
            policies = np.array([d[1] for d in training_data])
            values = np.array([d[2] for d in training_data])
            np.savez(filename, states=states, policies=policies, values=values)
            print(f"✓ Game saved to {filename} for AI training!")

        if result == "1-0":
            self.message = "Checkmate! White wins."
            print("\n🏆 CHECKMATE! White wins.")
        elif result == "0-1":
            self.message = "Checkmate! Black wins."
            print("\n🏆 CHECKMATE! Black wins.")
        else:
            self.message = "Draw!"
            print("\n🤝 Draw!")
        self.draw_board()


if __name__ == "__main__":
    # Ask the user what color they want to play
    human_color = choose_color()
    
    # Launch the main game window
    root = tk.Tk()
    gui = ChessGUI(root, human_color)
    root.mainloop()