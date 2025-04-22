import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import chess

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)
        self.observation_space = spaces.Box(low=0, high=1, shape=(768,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.board.reset()
        return self._get_observation(), {}

    def _get_observation(self):
        tensor = np.zeros((8, 8, 12), dtype=np.float32)
        piece_types = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                idx = piece_types.index(piece.symbol())
                tensor[chess.square_rank(square), chess.square_file(square), idx] = 1
        return tensor.flatten()

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        move = legal_moves[action] if action < len(legal_moves) else legal_moves[0]
        self.board.push(move)
        reward = (
            1 if self.board.is_checkmate()
            else 0 if self.board.is_stalemate() or self.board.is_insufficient_material()
            else -1 if self.board.is_game_over()
            else 0
        )
        done = self.board.is_game_over()
        return self._get_observation(), reward, done, False, {}

class ChessRLTrainer:
    def __init__(self, model_path="data/chess_rl_model.zip"):
        self.env = ChessEnv()
        self.model = PPO("MlpPolicy", self.env, verbose=0)
        self.model_path = model_path

    def train_on_game(self, game, timesteps=500):
        board = game.board()
        obs, _ = self.env.reset()
        for move in game.mainline_moves():
            action, _ = self.model.predict(obs)
            obs, reward, done, truncated, _ = self.env.step(action)
            if done or truncated:
                break
        self.model.learn(total_timesteps=timesteps)

    def save(self, name=None):
        path = name if name else self.model_path
        self.model.save(path)
