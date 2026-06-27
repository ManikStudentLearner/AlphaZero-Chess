♟️ AlphaZero-Chess: A Self-Learning AI
A from-scratch implementation of DeepMind's AlphaZero algorithm applied to Chess, built entirely with PyTorch and NumPy.

The AI starts with zero knowledge of chess strategy—only the rules of the game—and learns entirely through deep reinforcement learning and self-play. The longer it runs, the smarter it gets. It also features a continuous learning pipeline that allows it to learn from games played against humans.

✨ Key Features
Pure AlphaZero Architecture: Implements the full Residual CNN + Monte Carlo Tree Search (MCTS) pipeline.
Continuous Learning: The AI saves its brain and replay buffer. Every time you run train.py, it picks up exactly where it left off, getting stronger with every iteration.
Imitation Learning: Play against the AI in the GUI, and it saves your games. During its next training cycle, it imports your human games to learn from your strategies.
Interactive GUI: A sleek, click-to-play chessboard (built with Tkinter) where you can play as White or Black.
Live Training Observer: Watch the AI play against itself in real-time while it trains, with moves simultaneously printed to the terminal.
🧠 Architecture & Mathematics
The system is comprised of three core pillars: the Neural Network, the Tree Search, and the Self-Play Loop.

1. The Neural Network (The Intuition)
A Residual Convolutional Network takes the 18-channel board state as input and outputs two predictions:

Policy Head: A probability distribution over 4,672 possible moves (predicting which move to make).
Value Head: A scalar in [-1, 1] (predicting who is winning from the current player's perspective).
2. Monte Carlo Tree Search (The Calculation)
MCTS uses the neural network's predictions to guide its search. It balances exploration and exploitation using the Upper Confidence Bound (UCB) formula:

UCB(s, a) = -Q(s, a) + c_{puct} \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}
Where Q is the action-value, P is the prior probability from the policy head, N is the visit count, and c_puct is the exploration constant.

3. Board & Move Encoding
State Space: 18×8×8 tensor encoding piece positions (from the current player's perspective), castling rights, en passant squares, and side-to-move.
Action Space: 4,672-dim vector encoding Queen moves (8 directions × 7 distances), Knight moves (8 offsets), and Underpromotions (3 pieces × 3 directions) across 64 origin squares.

🚀 Quick Start
1. Installation

bash
git clone https://github.com/ManikStudentLearner/AlphaZero-Chess/
cd AlphaZero-Chess
pip install -r requirements.txt

3. Play Against the AI
Run the graphical interface and choose your color:

bash
python play_gui.py

Note: If no trained model exists, the AI will play randomly. Play a few games against it—the data is automatically saved for training!

3. Train the AI
Start the self-play training loop with the live observer GUI:

bash
python train.py

A window will pop up showing the AI playing against itself.
Moves are printed in UCI format (e.g., e2e4) in your terminal.
Press Ctrl+C or close the window to stop. Progress is saved automatically.

🔄 How It Learns (The Feedback Loop)
Self-Play: The current model plays games against itself using MCTS to select moves.
Data Collection: Every position is stored with the MCTS policy and the final game outcome.
Human Ingestion: Any games saved from play_gui.py are injected into the replay buffer.
Training: The network updates its weights to predict the MCTS policy (what to search) and the game outcome (who wins).
Iteration: The improved model plays better games, generating higher-quality training data, creating an upward spiral of intelligence.


🛠️ Tech Stack for nerds
Python 3.10+
PyTorch: Neural network architecture, autograd, and optimization.
python-chess: Legal move generation, board state, and UCI formatting.
NumPy: High-performance tensor operations for board encoding.
Tkinter: Built-in Python GUI framework for interactive play and live training observation.
text

