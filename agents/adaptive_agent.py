import chess
import numpy as np
from stable_baselines3 import PPO

class AdaptiveAgent:
    def __init__(self, model_path="data/chess_rl_model.zip"):
        """Cargar modelo PPO entrenado con stable-baselines3"""
        self.model = PPO.load(model_path)  # Solo cargamos el modelo PPO entrenado

    def encode_board(self, board):
        """Codificar el estado del tablero en un tensor de 8x8x12."""
        piece_types = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        tensor = np.zeros((8, 8, 12), dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = piece_types.index(piece.symbol())
                tensor[chess.square_rank(square), chess.square_file(square), piece_idx] = 1
        return tensor.flatten()

    def predict_move(self, board: chess.Board) -> chess.Move:
        """Predecir la jugada basada en el estado del tablero utilizando el modelo PPO."""
        # Obtener el estado del tablero codificado
        state = self.encode_board(board)

        # Predecir la acción con el modelo PPO
        action, _ = self.model.predict(np.array([state]), deterministic=True)

        legal_moves = list(board.legal_moves)

        # Asegurarse de que la acción es válida
        if action < len(legal_moves):
            return legal_moves[action]
        else:
            return legal_moves[0]  # Fallback si la acción no es válida

    def suggest_move(self, fen: str) -> str:
        """Devuelve la jugada adaptativa en formato UCI"""
        board = chess.Board(fen)
        move = self.predict_move(board)
        return move.uci()
