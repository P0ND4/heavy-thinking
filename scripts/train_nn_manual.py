import chess
import chess.pgn
import chess.engine
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import csv
import os

# Crear directorio si no existe
os.makedirs("data", exist_ok=True)

STOCKFISH_PATH = "C:\\Melvin\\stockfish\\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# 1. Cargar FENs ya evaluados
evaluated_fens = set()
csv_path = "data/chess_evaluations.csv"
if os.path.exists(csv_path):
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # saltar header
        for row in reader:
            if row:
                evaluated_fens.add(row[0])

# 2. Cargar partidas
games = []
pgn_dir = "data/database"
max_games = 5000
loaded_games = 0

for filename in os.listdir(pgn_dir):
    if not filename.endswith(".pgn"):
        continue
    
    with open(os.path.join(pgn_dir, filename), encoding="utf-8") as pgn_file:
        while loaded_games < max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
            loaded_games += 1
            
    if loaded_games >= max_games:
        break

# 3. Funciones auxiliares
def encode_board(board):
    piece_types = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_types.index(piece.symbol())
            tensor[chess.square_rank(square), chess.square_file(square), idx] = 1
    return tensor.flatten()

def evaluate_board(engine, board):
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=10))
        score = info["score"].white().score(mate_score=10000)
        if score is None:
            return 0
        return np.tanh(score / 100)
    except:
        return 0

# 4. Crear o continuar archivo CSV
csv_mode = "a" if os.path.exists(csv_path) else "w"
csv_file = open(csv_path, mode=csv_mode, newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
if csv_mode == "w":
    csv_writer.writerow(["fen", "evaluation"] + [f"feature_{i}" for i in range(768)])

# 5. Recolectar datos
X, y = [], []
for game in tqdm(games):
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if i % 2 != 0:  # evaluar cada 2 jugadas
            board.push(move)
            continue
        board.push(move)
        fen = board.fen()
        if fen in evaluated_fens:
            continue

        encoded = encode_board(board)
        evaluation = evaluate_board(engine, board)

        X.append(encoded)
        y.append(evaluation)
        csv_writer.writerow([fen, evaluation] + encoded.tolist())
        evaluated_fens.add(fen)

csv_file.close()
engine.quit()

# 6. Entrenar modelo con todo el dataset acumulado
import pandas as pd
df = pd.read_csv(csv_path)
X = df[[f"feature_{i}" for i in range(768)]].values
y = df["evaluation"].values

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(768,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation="tanh")
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)
model.save("data/nn_model.keras")
