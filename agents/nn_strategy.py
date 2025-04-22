import chess
import numpy as np
import tensorflow as tf

class NNStrategy:
    def __init__(self, model_path="data/nn_model.keras"):
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

    def encode_board(self, board):
        """Codificar el estado del tablero en un tensor de 8x8x12."""
        piece_types = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"] 
        tensor = np.zeros((8, 8, 12), dtype=np.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = piece_types.index(piece.symbol())
                tensor[chess.square_rank(square), chess.square_file(square), idx] = 1

        return tensor.flatten()

    def evaluate(self, fen: str) -> float:
        """Evaluar la posición dada en formato FEN utilizando el modelo."""
        # Crear el tablero a partir del FEN
        board = chess.Board(fen)
        
        # Codificar el tablero
        encoded = self.encode_board(board)

        # Predecir la evaluación usando el modelo
        evaluation = self.model.predict(np.array([encoded]))[0][0]
        
        return evaluation
    
    def suggest_move(self, fen: str) -> str:
        """Devuelve el mejor movimiento en formato UCI para un FEN dado"""
        board = chess.Board(fen)
        best_move = None
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            eval = self.evaluate(board.fen())
            board.pop()
            
            if (board.turn == chess.WHITE and eval > best_eval) or \
               (board.turn == chess.BLACK and eval < best_eval):
                best_eval = eval
                best_move = move.uci()
        
        return best_move if best_move else list(board.legal_moves)[0].uci()
    
    def select_best_move(self, board):
        legal_moves = list(board.legal_moves)
        best_score = -np.inf
        best_move = None

        for move in legal_moves:
            board.push(move)
            encoded = self.encode_board(board)
            score = self.model.predict(np.array([encoded]))[0][0]
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
