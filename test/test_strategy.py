import chess
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import NNStrategy from agents package
from agents.nn_strategy import NNStrategy

# Initialize the NNStrategy class
NN = NNStrategy()

# Test with initial position (standard chess starting position)
initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
initial_move = NN.select_best_move(chess.Board(initial_fen))
print(f"Suggested move (initial position): {initial_move}")

# Test with custom position
custom_fen = "rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq h3 0 1"
custom_move = NN.select_best_move(chess.Board(custom_fen))
print(f"Suggested move (custom position): {custom_move}")