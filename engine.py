# engine.py
import chess
import chess.pgn
import os
from datetime import datetime
from agents.nn_strategy import NNStrategy
from agents.adaptive_agent import AdaptiveAgent
from agents.meta_agent import MetaAgent
from config import config
from training.train_nn import ChessNNTrainer
from training.train_rl import ChessRLTrainer

class ChessEngine:
    def __init__(self, config):
        self.config = config
        self.nn_trainer = ChessNNTrainer(stockfish_path="C:\\Melvin\\stockfish\\stockfish-windows-x86-64-avx2.exe")
        self.rl_trainer = ChessRLTrainer()
        self.nn_strategy = NNStrategy(model_path=self.config['nn_model_path'])
        self.adaptive_agent = AdaptiveAgent(model_path=self.config['adaptive_model_path'])
        self.meta_agent = MetaAgent(self.nn_strategy, self.adaptive_agent, weights=self.config['agent_weights'])

    def play_game(self, game_id=0):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = "Auto-SelfPlay"
        game.headers["Date"] = datetime.today().strftime("%Y.%m.%d")
        game.headers["White"] = "MetaAgent"
        game.headers["Black"] = "MetaAgent"

        node = game
        while not board.is_game_over():
            move = self.meta_agent.suggest_move(board.fen())
            chess_move = chess.Move.from_uci(move)
            board.push(chess_move)
            print(f"Board: \n {chess.Board(board.fen())}")
            node = node.add_variation(chess_move)

        game.headers["Result"] = board.result()

        # Guardar la partida
        pgn_path = f"data/generated/game_{game_id}.pgn"
        os.makedirs("data/generated", exist_ok=True)
        with open(pgn_path, "w", encoding="utf-8") as f:
            print(game, file=f)

        return game, pgn_path

    def train_from_game(self, pgn_path):
        self.nn_trainer.train_from_game(pgn_path)
        self.rl_trainer.train_on_game(pgn_path)

    def run_loop(self):
        for i in range(3):
            print(f"\n▶️ Jugando partida #{i}")
            game, _ = self.play_game(i)
            print(f"✅ Partida guardada: {game}")

            print("🔧 Entrenando con la partida...")
            self.train_from_game(game)
            print("✅ Entrenamiento completo")

if __name__ == "__main__":
    engine = ChessEngine(config)
    engine.run_loop()
