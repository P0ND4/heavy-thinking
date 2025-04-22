import chess
from typing import Dict, Optional
from agents.nn_strategy import NNStrategy
from agents.adaptive_agent import AdaptiveAgent

class MetaAgent:
    def __init__(
        self,
        nn_agent: NNStrategy,
        adaptive_agent: AdaptiveAgent,
        weights: Dict[str, float] = None
    ):
        self.agents = {
            'nn': nn_agent,
            'adaptive': adaptive_agent
        }
        self.weights = weights or {
            'nn': 0.8,
            'adaptive': 0.7
        }

    def suggest_move(self, fen: str) -> str:
        """Combina inteligentemente las estrategias disponibles."""
        board = chess.Board(fen)
        
        if board.is_game_over():
            raise ValueError("La partida ya ha terminado")
        
        suggestions = self._get_all_suggestions(board)
        return self._select_best_move(board, suggestions)

    def _get_all_suggestions(self, board: chess.Board) -> Dict[str, str]:
        """Obtiene sugerencias de los agentes."""
        return {
            'nn': self._get_nn_move(board),
            'adaptive': self._get_adaptive_move(board)
        }

    def _get_nn_move(self, board: chess.Board) -> Optional[str]:
        """Obtiene movimiento del NNStrategy con verificación de seguridad."""
        move = self.agents['nn'].suggest_move(board.fen())
        return move
    
    def _get_adaptive_move(self, board: chess.Board) -> Optional[str]:
        """Obtiene movimiento del AdaptiveAgent con verificación de seguridad."""
        move = self.agents['adaptive'].suggest_move(board.fen())
        return move

    def _select_best_move(self, board: chess.Board, suggestions: Dict[str, str]) -> str:
        """Selección inteligente del mejor movimiento utilizando la estrategia del NN."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No hay movimientos legales disponibles")
        
        # Evaluar el movimiento propuesto por NN
        best_nn_move = self.agents['nn'].select_best_move(board)
        
        scored_moves = {}
        for move in legal_moves:
            score = 0.0
            move_uci = move.uci()
            
            # Añadir puntuación base para movimientos legales
            score += 0.1  

            # Añadir puntuación basada en los agentes
            for agent_name, suggested_move in suggestions.items():
                if suggested_move == move_uci:
                    score += self.weights[agent_name]
            
            # Añadir la puntuación del movimiento sugerido por NN
            if best_nn_move == move_uci:
                score += self.weights['nn']
            
            scored_moves[move_uci] = score
        
        # Elegir el movimiento con mayor puntuación
        return max(scored_moves.items(), key=lambda x: x[1])[0]

    def update_weights(self, new_weights: Dict[str, float]):
        """Actualización dinámica de pesos."""
        self.weights.update({
            k: v for k, v in new_weights.items() 
            if k in self.weights
        })
