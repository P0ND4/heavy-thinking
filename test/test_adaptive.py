import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import AdaptiveAgent from agents package
from agents.adaptive_agent import AdaptiveAgent

# Initialize the AdaptiveAgent
adaptive_agent = AdaptiveAgent(model_path="data/chess_rl_model")

# Test 1: Initial position (should suggest a reasonable opening move)
initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
initial_move = adaptive_agent.suggest_move(initial_fen)
print(f"Suggested move (initial position): {initial_move}")

# Test 2: Common opening position
common_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
common_move = adaptive_agent.suggest_move(common_fen)
print(f"Suggested move (common opening position): {common_move}")

# Test 3: Tactical position (should find a good capture)
tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"
tactical_move = adaptive_agent.suggest_move(tactical_fen)
print(f"Suggested move (tactical position): {tactical_move}")

# Test 4: Endgame position
endgame_fen = "8/8/8/4k3/8/8/4K3/8 w - - 0 1"
endgame_move = adaptive_agent.suggest_move(endgame_fen)
print(f"Suggested move (endgame position): {endgame_move}")