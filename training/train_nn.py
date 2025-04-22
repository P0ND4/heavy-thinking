import numpy as np
import tensorflow as tf
import chess
import chess.engine

class ChessNNTrainer:
    def __init__(self, model_path="data/nn_model.keras", stockfish_path=None):
        self.model_path = model_path
        self.model = self._build_model()
        self.engine = None
        if stockfish_path:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(768,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation="tanh")
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def encode_board(self, board):
        piece_types = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = piece_types.index(piece.symbol())
                tensor[chess.square_rank(square), chess.square_file(square), idx] = 1
        return tensor.flatten()

    def evaluate_board(self, board):
        if not self.engine:
            return 0
        info = self.engine.analyse(board, chess.engine.Limit(depth=15))
        score = info["score"].white().score(mate_score=10000)
        return np.tanh(score / 100) if score else 0

    def train_from_game(self, game):
        board = game.board()
        X, y = [], []
        for move in game.mainline_moves():
            board.push(move)
            encoded = self.encode_board(board)
            evaluation = self.evaluate_board(board)
            X.append(encoded)
            y.append(evaluation)
        if X:
            self.model.fit(np.array(X), np.array(y), epochs=1, batch_size=16)
            self.model.save(self.model_path)

    def close(self):
        if self.engine:
            self.engine.quit()
    
    def play_game_against_itself(self, max_moves=100):
        board = chess.Board()
        X, y = [], []

        for _ in range(max_moves):
            if board.is_game_over():
                break

            move = self.select_best_move(board)
            if move is None:
                break

            board.push(move)
            encoded = self.encode_board(board)
            evaluation = self.evaluate_board(board)  # usa stockfish
            X.append(encoded)
            y.append(evaluation)

        if X:
            self.model.fit(np.array(X), np.array(y), epochs=10, batch_size=16)
            self.model.save(self.model_path)


