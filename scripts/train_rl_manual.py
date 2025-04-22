import chess
import chess.pgn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tqdm import tqdm

# 1. Cargar partidas PGN
pgn = open("data/games.pgn")  # Cambia por la ruta a tu archivo PGN
games = []

for _ in range(50):
    game = chess.pgn.read_game(pgn)
    if game is not None:
        games.append(game)

# 2. Definir el entorno personalizado
class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # Movimientos posibles
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(768,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.board.reset()
        return self._get_observation(), {}

    def _get_observation(self):
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        piece_types = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_idx = piece_types.index(piece.symbol())
                tensor[chess.square_rank(square), chess.square_file(square), piece_idx] = 1
        return tensor.flatten()

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if action < len(legal_moves):
            move = legal_moves[action]
        else:
            move = legal_moves[0]  # Acción inválida, se juega la primera legal
        self.board.push(move)

        if self.board.is_checkmate():
            reward = 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0
        elif self.board.is_game_over():
            reward = -1
        else:
            reward = 0

        done = self.board.is_game_over()
        return self._get_observation(), reward, done, False, {}

# 3. Verificar el entorno
env = ChessEnv()
check_env(env, warn=True)

# 4. Crear el modelo PPO
model = PPO("MlpPolicy", env, verbose=0)

# 5. Entrenamiento personalizado usando las partidas PGN
print("Entrenando agente...")

for i in tqdm(range(2), desc="Progreso de entrenamiento"):
    game = games[i % len(games)]
    board = game.board()
    obs, _ = env.reset()

    for move in game.mainline_moves():
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    model.learn(total_timesteps=500)

    if (i + 1) % 100 == 0:
        model.save(f"data/chess_rl_model_{i+1}.zip")
        print(f"Modelo guardado en data/chess_rl_model_{i+1}.zip")

# 6. Guardar modelo final
model.save("data/chess_rl_model.zip")
print("✅ Entrenamiento finalizado. Modelo guardado.")
