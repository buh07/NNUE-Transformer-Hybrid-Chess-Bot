"""
Time Management for Chess Bot
Dynamically allocates time per move based on position complexity and game state.
"""

import chess
import torch
from typing import Dict, Optional
from utils.chess_utils import extract_selection_features
import config


class TimeManager:
    """
    Dynamically allocates time per move based on:
    1. Selector confidence (how certain the model is about complexity)
    2. Position characteristics (tactical vs strategic)
    3. Game phase (opening/middlegame/endgame)
    4. Time remaining
    5. Move number
    """
    
    def __init__(self, 
                 total_time: float,
                 increment: float = 0.0,
                 moves_to_go: Optional[int] = None,
                 emergency_time: float = 5.0):
        """
        Args:
            total_time: Total time remaining (seconds)
            increment: Time increment per move (seconds)
            moves_to_go: Expected moves until time control (None = guess from position)
            emergency_time: Reserve time to keep for emergencies (seconds)
        """
        self.total_time = total_time
        self.increment = increment
        self.moves_to_go = moves_to_go
        self.emergency_time = emergency_time
        self.move_count = 0
        
        # Time allocation history for adaptation
        self.time_used_history = []
        self.selector_confidence_history = []
    
    def allocate_time(self, 
                     board: chess.Board,
                     selector_model,
                     depth_remaining: int = 10) -> Dict[str, float]:
        """
        Allocate time for current move based on position complexity
        
        Args:
            board: Current chess position
            selector_model: SelectionFunction model to assess complexity
            depth_remaining: Depth in search tree
        
        Returns:
            dict with:
                - 'allocated_time': Recommended time for this move
                - 'min_time': Minimum time to spend
                - 'max_time': Maximum time to spend
                - 'complexity_score': Position complexity [0-1]
                - 'selector_confidence': How confident selector is [0-1]
                - 'time_pressure': Time pressure factor [0-1]
        """
        self.move_count += 1
        
        # Get position features and selector probability
        features = extract_selection_features(board, depth_remaining)
        
        # Move features to same device as selector model
        device = next(selector_model.parameters()).device
        features = features.to(device)
        
        with torch.no_grad():
            selector_prob = selector_model.forward(features).item()
        
        # Selector confidence: how far from 0.5 (uncertain) to 0 or 1 (certain)
        selector_confidence = abs(selector_prob - 0.5) * 2.0  # 0 = uncertain, 1 = certain
        self.selector_confidence_history.append(selector_confidence)
        
        # Extract position characteristics from features
        complexity_metrics = self._extract_complexity_from_features(features, selector_prob)
        
        # Estimate moves to go if not specified
        if self.moves_to_go is None:
            estimated_moves = self._estimate_moves_remaining(board, complexity_metrics)
        else:
            estimated_moves = self.moves_to_go
        
        # Calculate base time allocation
        available_time = max(0, self.total_time - self.emergency_time)
        base_time = available_time / max(estimated_moves, 1) + self.increment * 0.8
        
        # Complexity multiplier (1.0 = normal, 2.0 = very complex, 0.5 = simple)
        complexity_multiplier = self._calculate_complexity_multiplier(complexity_metrics)
        
        # Uncertainty multiplier (uncertain positions need more time)
        uncertainty_multiplier = 1.0 + (1.0 - selector_confidence) * 0.5
        
        # Phase multiplier (critical phases need more time)
        phase_multiplier = self._calculate_phase_multiplier(board, complexity_metrics)
        
        # Time pressure factor (spend less when low on time)
        time_pressure = self._calculate_time_pressure()
        
        # Combined allocation
        allocated_time = (base_time * 
                         complexity_multiplier * 
                         uncertainty_multiplier * 
                         phase_multiplier * 
                         time_pressure)
        
        # Set bounds
        min_time = max(0.1, base_time * 0.3)  # At least 30% of base
        max_time = min(available_time * 0.2, base_time * 3.0)  # At most 20% of remaining
        allocated_time = max(min_time, min(allocated_time, max_time))
        
        return {
            'allocated_time': allocated_time,
            'min_time': min_time,
            'max_time': max_time,
            'base_time': base_time,
            'complexity_score': complexity_metrics['overall_complexity'],
            'selector_confidence': selector_confidence,
            'selector_probability': selector_prob,
            'time_pressure': time_pressure,
            'estimated_moves_remaining': estimated_moves,
            'multipliers': {
                'complexity': complexity_multiplier,
                'uncertainty': uncertainty_multiplier,
                'phase': phase_multiplier,
                'time_pressure': time_pressure
            }
        }
    
    def _extract_complexity_from_features(self, features: torch.Tensor, 
                                         selector_prob: float) -> Dict:
        """
        Extract complexity metrics from selection features
        
        Selection features (from chess_utils.py):
        [0] = move_count / 200
        [1] = material_balance / 40
        [2] = num_legal_moves / 50
        [3] = is_check
        [4] = num_attackers / 10
        [5] = num_defenders / 10
        [6] = center_control / 8
        [7] = mobility / 50
        [8] = king_safety / 10
        [9] = piece_activity / 16
        [10] = pawn_structure_score / 10
        [11] = is_endgame
        [12] = has_passed_pawns
        [13] = king_tropism / 50
        [14] = depth_remaining / 10
        [15] = num_pieces / 32
        [16] = bishops_vs_knights
        [17] = queen_on_board
        [18] = rooks_on_open_files / 4
        [19] = connected_rooks
        """
        feat = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
        
        # Tactical complexity (captures, checks, threats)
        tactical_complexity = (
            feat[3] * 0.3 +  # is_check
            feat[4] * 0.1 +  # num_attackers
            feat[5] * 0.1 +  # num_defenders
            min(feat[2], 1.0) * 0.3 +  # many legal moves
            (1.0 - feat[11]) * 0.2  # not endgame
        )
        
        # Strategic complexity (positional factors)
        strategic_complexity = (
            feat[6] * 0.2 +  # center_control
            feat[9] * 0.2 +  # piece_activity
            feat[10] * 0.2 +  # pawn_structure
            feat[13] * 0.2 +  # king_tropism
            feat[18] * 0.2  # rooks_on_open_files
        )
        
        # Material imbalance (unbalanced = more complex)
        material_imbalance = abs(feat[1]) * 2.0  # material_balance
        
        # Phase (opening/middlegame/endgame)
        move_count_norm = feat[0]  # already normalized by 200
        num_pieces_norm = feat[15]  # normalized by 32
        is_opening = 1.0 if move_count_norm < 0.1 else 0.0  # < 20 moves
        is_middlegame = 1.0 if (move_count_norm >= 0.1 and 
                                move_count_norm < 0.25 and 
                                num_pieces_norm > 0.4) else 0.0
        is_endgame = float(feat[11])
        
        # Overall complexity (higher selector_prob = more strategic = more complex)
        overall_complexity = (
            tactical_complexity * 0.3 +
            strategic_complexity * 0.3 +
            material_imbalance * 0.2 +
            selector_prob * 0.2  # Selector says use transformer = complex
        )
        
        return {
            'tactical_complexity': float(tactical_complexity),
            'strategic_complexity': float(strategic_complexity),
            'material_imbalance': float(material_imbalance),
            'overall_complexity': float(min(overall_complexity, 1.0)),
            'is_opening': is_opening,
            'is_middlegame': is_middlegame,
            'is_endgame': is_endgame,
            'move_count': feat[0] * 200,  # denormalize
            'num_pieces': feat[15] * 32,
            'num_legal_moves': feat[2] * 50,
            'is_check': bool(feat[3]),
            'selector_wants_transformer': selector_prob > config.SELECTOR_THRESHOLD
        }
    
    def _calculate_complexity_multiplier(self, complexity_metrics: Dict) -> float:
        """
        Calculate time multiplier based on position complexity
        
        Returns multiplier in range [0.5, 2.0]
        """
        complexity = complexity_metrics['overall_complexity']
        
        # Base multiplier from complexity score
        base_mult = 0.5 + complexity * 1.5  # 0.5 to 2.0
        
        # Bonus for critical positions
        if complexity_metrics['is_check']:
            base_mult *= 1.2
        
        if complexity_metrics['material_imbalance'] > 0.5:
            base_mult *= 1.1
        
        return min(base_mult, 2.0)
    
    def _calculate_phase_multiplier(self, board: chess.Board, 
                                   complexity_metrics: Dict) -> float:
        """
        Calculate time multiplier based on game phase
        
        Opening: 0.7x (use book knowledge, play fast)
        Middlegame: 1.2x (critical phase)
        Endgame: 1.0x (important but more calculation)
        """
        if complexity_metrics['is_opening']:
            return 0.7
        elif complexity_metrics['is_middlegame']:
            return 1.2
        elif complexity_metrics['is_endgame']:
            return 1.0
        else:
            return 1.0
    
    def _calculate_time_pressure(self) -> float:
        """
        Calculate time pressure factor
        
        Returns multiplier in range [0.5, 1.0]
        Low time = spend less per move
        """
        available_time = max(0, self.total_time - self.emergency_time)
        
        if available_time < 10:
            return 0.5  # Very low time
        elif available_time < 30:
            return 0.7  # Low time
        elif available_time < 60:
            return 0.85  # Medium time
        else:
            return 1.0  # Plenty of time
    
    def _estimate_moves_remaining(self, board: chess.Board, 
                                  complexity_metrics: Dict) -> int:
        """
        Estimate moves until game end or time control
        """
        move_count = complexity_metrics['move_count']
        
        if complexity_metrics['is_opening']:
            # Opening: expect ~60 more moves
            return max(60 - int(move_count), 40)
        elif complexity_metrics['is_middlegame']:
            # Middlegame: expect ~40 more moves
            return max(80 - int(move_count), 30)
        else:
            # Endgame: expect ~20-30 more moves
            num_pieces = complexity_metrics['num_pieces']
            if num_pieces < 10:
                return max(100 - int(move_count), 10)
            else:
                return max(90 - int(move_count), 20)
    
    def update_after_move(self, time_used: float):
        """Update time manager after move is made"""
        self.total_time -= time_used
        self.time_used_history.append(time_used)
        
        if self.moves_to_go is not None:
            self.moves_to_go -= 1
    
    def get_statistics(self) -> Dict:
        """Get time management statistics"""
        if not self.time_used_history:
            return {
                'total_time_remaining': self.total_time,
                'moves_played': self.move_count,
                'avg_time_per_move': 0.0,
                'avg_selector_confidence': 0.0
            }
        
        return {
            'total_time_remaining': self.total_time,
            'moves_played': self.move_count,
            'total_time_used': sum(self.time_used_history),
            'avg_time_per_move': sum(self.time_used_history) / len(self.time_used_history),
            'min_time_used': min(self.time_used_history),
            'max_time_used': max(self.time_used_history),
            'avg_selector_confidence': (sum(self.selector_confidence_history) / 
                                       len(self.selector_confidence_history)),
            'emergency_time_reserve': self.emergency_time,
            'available_time': max(0, self.total_time - self.emergency_time)
        }
    
    def should_stop_search(self, search_start_time: float, 
                          allocated_time: float,
                          min_time: float,
                          current_time: float) -> bool:
        """
        Decide if search should stop based on time
        
        Args:
            search_start_time: When search started
            allocated_time: Target time for this move
            min_time: Minimum time to search
            current_time: Current time
        
        Returns:
            True if search should stop
        """
        elapsed = current_time - search_start_time
        
        # Always search at least min_time
        if elapsed < min_time:
            return False
        
        # Stop if we've used allocated time
        if elapsed >= allocated_time:
            return True
        
        # Stop if we're in time trouble
        if self.total_time < self.emergency_time * 2:
            return elapsed >= min_time
        
        return False
