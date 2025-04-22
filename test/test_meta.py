import sys
from pathlib import Path
import chess

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import MetaAgent from agents package
from agents.meta_agent import MetaAgent

# Initialize the MetaAgent
meta_agent = MetaAgent()

# Test with initial position (standard chess starting position)
initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
initial_move = meta_agent.suggest_move(initial_fen)
print(f"Suggested move (initial position): {initial_move}")

# Test with a tactical position
tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"
tactical_move = meta_agent.suggest_move(tactical_fen)
print(f"Suggested move (tactical position): {tactical_move}")
print(f"Is move safe? {meta_agent._is_move_safe(chess.Board(tactical_fen), tactical_move)}")

# Test a known unsafe move
safety_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
unsafe_move = "f3e5"  # Knight captures defended pawn
print(f"Is move '{unsafe_move}' safe? {meta_agent._is_move_safe(chess.Board(safety_fen), unsafe_move)}")