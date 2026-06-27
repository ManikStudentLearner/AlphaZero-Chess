"""
Hyperparameters and configuration for AlphaZero Chess.
"""

# ─── Model Architecture ────────────────────────────────────
NUM_RES_BLOCKS      = 4       # Residual blocks in the tower
NUM_FILTERS         = 64      # Convolutional filters per block
INPUT_CHANNELS      = 18      # Board encoding channels
POLICY_OUTPUT_SIZE  = 4672    # 64 squares × 73 move types

# ─── MCTS ──────────────────────────────────────────────────
MCTS_SIMULATIONS    = 200     # Simulations per move (training)
MCTS_SIMULATIONS_PLAY = 400   # Simulations per move (playing)
C_PUCT              = 1.41    # Exploration constant
DIRICHLET_ALPHA     = 0.3     # Noise alpha for chess
DIRICHLET_EPSILON   = 0.25    # Noise mixing weight

# ─── Self-Play ─────────────────────────────────────────────
TEMPERATURE         = 1.0     # Temperature for early moves
TEMPERATURE_THRESHOLD = 10    # Moves before dropping temperature
MAX_MOVES           = 200     # Max moves before declaring draw

# ─── Training ──────────────────────────────────────────────
NUM_ITERATIONS      = 50      # Outer training iterations
GAMES_PER_ITERATION = 10      # Self-play games per iteration
EPOCHS_PER_ITERATION = 2      # Training epochs per iteration
BATCH_SIZE          = 256     # Mini-batch size
LEARNING_RATE       = 0.001   # Adam learning rate
WEIGHT_DECAY        = 1e-4    # L2 regularization
BUFFER_CAPACITY     = 100000  # Replay buffer size