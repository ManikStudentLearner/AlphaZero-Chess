"""
Monte Carlo Tree Search guided by the AlphaZero neural network.

UCB selection formula:
    UCB(s, a) = -Q(child) + c_puct · P(a|s) · √N(s) / (1 + N(s, a))

Values are stored from each node's *own* player perspective;
the negation in UCB converts the child's view to the parent's.
"""

import chess
import torch
import numpy as np
from environment import encode_board, get_game_value
from move_encoder import move_to_index, index_to_move, ACTION_SIZE
from config import C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON


def _softmax(x: np.ndarray) -> np.ndarray:
    # Safety check for empty arrays (e.g., terminal states with 0 legal moves)
    if x.size == 0:
        return x
    e = np.exp(x - np.max(x))
    return e / e.sum()


class MCTSNode:
    """A single node in the search tree."""

    __slots__ = (
        "move", "parent", "children", "prior",
        "visit_count", "value_sum", "is_expanded",
    )

    def __init__(self, move: chess.Move | None = None, parent=None, prior: float = 0.0):
        self.move = move
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def q_value(self) -> float:
        """Average value from this node's own player perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def select_child(self) -> "MCTSNode":
        """Pick the child maximizing UCB."""
        best_score = -float("inf")
        best_child = None
        total_parent_visits = self.visit_count

        for child in self.children:
            q = -child.q_value()  # Negate: child's view → parent's view
            u = (C_PUCT * child.prior
                 * np.sqrt(total_parent_visits) / (1 + child.visit_count))
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy_logits: np.ndarray, legal_moves: list[chess.Move],
               board: chess.Board):
        """
        Create one child per legal move. Policy logits are softmaxed
        *only over legal moves* so probability mass is not wasted.
        """
        self.is_expanded = True

        # Gather logit scores for legal moves only
        legal_indices = [move_to_index(m) for m in legal_moves]
        legal_logits = policy_logits[legal_indices]
        legal_probs = _softmax(legal_logits)

        for move, prob in zip(legal_moves, legal_probs):
            child = MCTSNode(move=move, parent=self, prior=prob)
            self.children.append(child)

    def backpropagate(self, value: float):
        """Walk up the tree, flipping the sign at every level."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip perspective for the parent
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search with neural-network guidance.

    Usage:
        mcts = MCTS(model, device)
        probs = mcts.search(board, num_simulations=200)
        best_move = mcts.pick_move(board, probs)
    """

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def _evaluate(self, board: chess.Board):
        """
        Run the neural network on the current board position.

        Returns:
            (full_policy_logits, value) — logits are shape (4672,).
        """
        state = encode_board(board)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(state_tensor)
        return policy_logits.cpu().numpy()[0], value.item()

    def search(self, board: chess.Board, num_simulations: int,
               add_noise: bool = False) -> np.ndarray:
        """
        Run MCTS from the given board position.

        Args:
            board: Current chess.Board state.
            num_simulations: Number of playouts to run.
            add_noise: If True, inject Dirichlet noise at the root.

        Returns:
            action_probs: shape (4672,) — visit-count probability distribution.
        """
        root = MCTSNode()

        # Evaluate root once to expand it
        policy_logits, root_value = self._evaluate(board)
        legal_moves = list(board.legal_moves)
        root.expand(policy_logits, legal_moves, board)

        # Optional Dirichlet noise for exploration during self-play
        if add_noise and root.children:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior = (1 - DIRICHLET_EPSILON) * child.prior + DIRICHLET_EPSILON * noise[i]

        # ── Run simulations ────────────────────────────────
        for _ in range(num_simulations):
            node = root
            sim_board = board.copy()

            # 1. Selection — walk down the tree
            while not node.is_leaf():
                node = node.select_child()
                sim_board.push(node.move)

            # 2. Expansion & Evaluation
            if sim_board.is_game_over():
                value = get_game_value(sim_board)
            else:
                logits, value = self._evaluate(sim_board)
                legal = list(sim_board.legal_moves)
                node.expand(logits, legal, sim_board)

            # 3. Backpropagation
            node.backpropagate(value)

        # ── Build visit-count policy ───────────────────────
        action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        for child in root.children:
            idx = move_to_index(child.move)
            action_probs[idx] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def pick_move(self, board: chess.Board, action_probs: np.ndarray,
                  temperature: float = 1.0) -> chess.Move:
        """
        Sample a move from the policy. Low temperature → near-deterministic.
        """
        if temperature < 1e-3:
            # Greedy: pick the most-visited move
            best_idx = np.argmax(action_probs)
            move = index_to_move(best_idx, board)
            # Fallback: if encoding mapping fails for any reason, pick the first legal move
            if move is None or move not in board.legal_moves:
                return list(board.legal_moves)[0]
            return move

        # Apply temperature
        probs = np.power(action_probs, 1.0 / temperature)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            # If all probabilities collapsed to 0, uniform over legal moves
            legal_indices = [move_to_index(m) for m in board.legal_moves]
            probs = np.zeros(ACTION_SIZE, dtype=np.float32)
            probs[legal_indices] = 1.0 / len(legal_indices)

        chosen_idx = np.random.choice(ACTION_SIZE, p=probs)
        move = index_to_move(chosen_idx, board)
        
        # Fallback safety net
        if move is None or move not in board.legal_moves:
            return list(board.legal_moves)[0]
            
        return move