"""
Connect-6 game engine with alpha-beta pruning search.

This module implements a SearchEngine class for playing Connect-6 using
alpha-beta pruning with transposition tables, Zobrist hashing, and advanced
move ordering techniques.
"""

from tools import *
import random
import numpy as np
import numba
from collections import OrderedDict
# 'influence': 0.4833960258781689, 'pattern': 0.5166039741218311

class SearchEngine():
    """
    Game tree search engine using alpha-beta pruning for Connect-6.
    
    This class implements an optimized minimax search with alpha-beta pruning,
    transposition tables, Zobrist hashing, killer move heuristics, and
    quiescence search for tactical positions.
    
    Attributes
    ----------
    m_board : list or None
        Current game board state
    m_chess_type : int or None
        Type of chess piece
    m_alphabeta_depth : int or None
        Maximum search depth for alpha-beta pruning
    m_total_nodes : int
        Total number of nodes explored
    last_positions : tuple or None
        Last move positions
    stone_count : int
        Number of stones on the board
    weights : dict
        Evaluation function weights for influence and pattern scoring
    transposition_table : OrderedDict
        Cache for previously evaluated positions
    max_table_size : int
        Maximum number of entries in transposition table
    zobrist_table : dict
        Zobrist hash values for incremental hashing
    current_hash : int
        Current board position hash
    metrics : dict
        Performance metrics for search diagnostics
    killer_moves : dict
        Killer move heuristic storage by depth
    """

    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_alphabeta_depth = None
        self.m_total_nodes = 0
        self.last_positions = None
        self.stone_count = 0
        # --- weights for evaluation (evolvable) ---
        self.weights = {
            "influence": 0.4833960258781689,
            "pattern": 0.5166039741218311
        }
        # --- initialize cache for transposition table ---
        self.transposition_table = OrderedDict()
        self.max_table_size = 500000

        self.zobrist_table = self.init_zobrist_table()
        self.current_hash = 0

        self.metrics = {
            'nodes_expanded': 0,
            'nodes_pruned': 0,
            'transposition_hits': 0,
            'quiescence_calls': 0,
            'max_depth_reached': 0,
            'decision_time': 0,
            'min_depth_seen': float('inf'),
            'initial_depth': 0
        }

        self.killer_moves = {}
    
    def reset_metrics(self):
        """
        Reset all metrics before a new search.
        
        Clears counters for nodes expanded, pruned, transposition hits,
        and other diagnostic information.
        """
        self.metrics = {
            'nodes_expanded': 0,
            'nodes_pruned': 0,
            'transposition_hits': 0,
            'quiescence_calls': 0,
            'max_depth_reached': 0,
            'decision_time': 0,
            'min_depth_seen': float('inf'),
            'initial_depth': 0
        }

    def init_zobrist_table(self):
        """
        Initialize Zobrist hash table for incremental position hashing.
        
        Each cell (x,y) and each possible stone type (BLACK, WHITE)
        gets a random 64-bit integer for fast position hashing.
        
        Returns
        -------
        dict
            Zobrist table mapping (x, y, color) to 64-bit hash values
        """
        random.seed(2024)  # fixed seed for reproducibility
        table = {}
        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                # Random numbers for BLACK and WHITE stones
                table[(x, y, Defines.BLACK)] = random.getrandbits(64)
                table[(x, y, Defines.WHITE)] = random.getrandbits(64)
        return table

    def compute_board_hash(self, board):
        """
        Compute Zobrist hash from scratch for current board position.
        
        Parameters
        ----------
        board : list of list
            Current board state
            
        Returns
        -------
        int
            64-bit Zobrist hash of the position
        """
        h = 0
        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                cell = board[x][y]
                # Only hash real stones
                if cell not in (Defines.BLACK, Defines.WHITE):
                    continue
                h ^= self.zobrist_table[(x, y, cell)]
        return h

    def update_board_hash(self, x, y, color):
        """
        Incrementally update hash when placing or removing a stone.
        
        Calling this twice with same (x,y,color) restores original hash.
        
        Parameters
        ----------
        x : int
            Row coordinate
        y : int
            Column coordinate
        color : int
            Stone color (BLACK or WHITE)
        """
        if color not in (Defines.BLACK, Defines.WHITE):
            return
        self.current_hash ^= self.zobrist_table[(x, y, color)]

    def alpha_beta_pruning(self, board, depth, alpha, beta, maximizing_player, last_move, max_candidates=40, is_root=False):
        """
        Perform alpha-beta pruning search to find best move.
        
        Implements minimax search with alpha-beta pruning, transposition tables,
        killer move heuristics, and quiescence search for tactical positions.
        
        Parameters
        ----------
        board : list of list
            Current board state
        depth : int
            Remaining search depth
        alpha : float
            Best score for maximizing player
        beta : float
            Best score for minimizing player
        maximizing_player : bool
            True if current player is maximizing
        last_move : StoneMove or tuple
            Last move played
        max_candidates : int, optional
            Maximum candidate moves to consider (default: 40)
        is_root : bool, optional
            True if this is the root of the search tree (default: False)
            
        Returns
        -------
        tuple
            (score, best_move) where score is evaluation and best_move is StoneMove
        """
        
        self.np_board = np.array(board)
        if is_root:
            self.reset_metrics()
            self.current_hash = self.compute_board_hash(board)
            self.transposition_table.clear()
            self.metrics['initial_depth'] = depth
            self.metrics['min_depth_seen'] = depth
        
        # Track metrics
        self.metrics['nodes_expanded'] += 1
        self.metrics['min_depth_seen'] = min(self.metrics['min_depth_seen'], depth)
        self.metrics['max_depth_reached'] = self.metrics['initial_depth'] - self.metrics['min_depth_seen']
        
        # Check transposition table
        state_key = self.current_hash
        if not is_root and state_key in self.transposition_table:
            cached_depth, cached_score, cached_move, cached_flag = self.transposition_table[state_key]
            
            if cached_depth >= depth:
                self.metrics['transposition_hits'] += 1
                
                if cached_flag == 'EXACT':
                    return (cached_score, cached_move)
                elif cached_flag == 'LOWERBOUND':
                    alpha = max(alpha, cached_score)
                elif cached_flag == 'UPPERBOUND':
                    beta = min(beta, cached_score)
                
                if alpha >= beta:
                    return (cached_score, cached_move)

        # Terminal conditions
        result = check_game_result(board, last_move)
        if result == Defines.BLACK:
            return (Defines.MAXINT, None)
        elif result == Defines.WHITE:
            return (Defines.MININT, None)
        elif result == Defines.DRAW:
            return (0, None)

        if depth == 0:
            q_score = self.quiescence_search(alpha, beta, maximizing_player, last_move)
            return (q_score, None)

        # Check immediate threats
        threats = numpy_immediate_threats(self.np_board, Defines.BLACK if maximizing_player else Defines.WHITE, Defines.BLACK, Defines.WHITE, Defines.NOSTONE)
        if len(threats) > 1:
            best_threat, second_threat = threats[0], threats[1]
            best_move = StoneMove()
            best_move.positions = [
                StonePosition(best_threat[0], best_threat[1]),
                StonePosition(second_threat[0], second_threat[1])
            ]
            return (Defines.MAXINT // 2, best_move)

        # Generate and order candidates
        if depth <= 2:
            effective_candidates = 15  # Even fewer when deep
        else:
            effective_candidates = min(max_candidates, 30)
        
        singles = self.generate_candidate_moves(board, last_move, effective_candidates)
        candidates = generate_candidate_pairs(singles, max_pairs=20)  
        
        if not candidates:
            return (self.evaluate_board(board, last_move), None)
        
        # ✅ Order moves BEFORE searching
        candidates = self.order_moves(board, candidates)

        if depth in self.killer_moves:
            candidates = self.killer_moves[depth] + candidates

        best_move = None
        value = -float("inf") if maximizing_player else float("inf")

        for move in candidates:
            pos1 = move.positions[0]
            pos2 = move.positions[1]
            color = Defines.BLACK if maximizing_player else Defines.WHITE

            # Apply move
            board[pos1.x][pos1.y] = color
            self.update_board_hash(pos1.x, pos1.y, color)
            if not (pos2.x == 0 and pos2.y == 0):
                board[pos2.x][pos2.y] = color
                self.update_board_hash(pos2.x, pos2.y, color)

            # Recursive call - ALWAYS returns 2 values
            eval_score, _ = self.alpha_beta_pruning(
                board, depth - 1, alpha, beta, not maximizing_player, move, max_candidates, is_root=False
            )
            
            # Undo move
            board[pos1.x][pos1.y] = Defines.NOSTONE
            self.update_board_hash(pos1.x, pos1.y, color)
            if not (pos2.x == 0 and pos2.y == 0):
                board[pos2.x][pos2.y] = Defines.NOSTONE
                self.update_board_hash(pos2.x, pos2.y, color)

            # Update best
            if maximizing_player:
                if eval_score > value:
                    value = eval_score
                    best_move = move
                alpha = max(alpha, value)
            else:
                if eval_score < value:
                    value = eval_score
                    best_move = move
                beta = min(beta, value)

            # Pruning
            if beta <= alpha:
                self.killer_moves[depth] = [move] + self.killer_moves.get(depth, [])[:1]
                self.metrics['nodes_pruned'] += 1
                break
        
        # ✅ Store in transposition table AFTER loop
        flag = 'EXACT'
        if value <= alpha:
            flag = 'UPPERBOUND'
        elif value >= beta:
            flag = 'LOWERBOUND'
        
        if state_key in self.transposition_table:
            self.transposition_table.move_to_end(state_key)
        self.transposition_table[state_key] = (depth, value, best_move, flag)
        if len(self.transposition_table) > self.max_table_size:
            self.transposition_table.popitem(last=False)  # remove LRU/oldest

        return value, best_move  # ✅ ALWAYS 2 values
    
    def generate_candidate_moves(self, board, last_move=None, max_candidates=15, radius=3):
        """
        Generate candidate move positions based on occupied stones.
        
        Creates a set of empty positions within a radius of occupied stones
        and last move, prioritizing tactical positions near recent activity.
        
        Parameters
        ----------
        board : list of list
            Current board state
        last_move : StoneMove or tuple or list, optional
            Last move(s) played (default: None)
        max_candidates : int, optional
            Maximum number of candidates to return (default: 15)
        radius : int, optional
            Search radius around occupied stones (default: 3)
            
        Returns
        -------
        list of tuple
            List of (x, y) coordinates sorted by tactical value
        """
        candidates = set()
        # All occupied stones
        occupied = [
            (x, y) for x in range(Defines.GRID_NUM)
                    for y in range(Defines.GRID_NUM)
                    if board[x][y] != Defines.NOSTONE
        ]

        if not occupied:
            return [(Defines.GRID_NUM // 2, Defines.GRID_NUM // 2)]

        # --- Normalize last_move into a list of (x, y) coordinates ---
        last_positions = []
        if last_move is not None:
            if isinstance(last_move, StoneMove):   # Case: StoneMove
                for pos in last_move.positions:
                    # filter dummy (0,0) if unused
                    if board[pos.x][pos.y] != Defines.NOSTONE or (pos.x, pos.y) != (0, 0):
                        last_positions.append((pos.x, pos.y))
            elif isinstance(last_move, list):      # Case: list
                for mv in last_move:
                    if isinstance(mv, StonePosition):
                        last_positions.append((mv.x, mv.y))
                    elif isinstance(mv, tuple) and len(mv) == 2:
                        last_positions.append(mv)
            elif isinstance(last_move, tuple) and len(last_move) == 2:  # single tuple
                last_positions.append(last_move)

        # Expand neighborhood around last positions
        for (lx, ly) in last_positions:
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = lx+dx, ly+dy
                    if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM:
                        if board[nx][ny] == Defines.NOSTONE:
                            candidates.add((nx, ny))

        # Expand around all occupied stones
        for (x, y) in occupied:
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM:
                        if board[nx][ny] == Defines.NOSTONE:
                            candidates.add((nx, ny))
        def move_score(mv):
            """
            Compute heuristic score for a candidate move position.
            
            Parameters
            ----------
            mv : tuple
                (x, y) coordinate of candidate position
                
            Returns
            -------
            float
                Heuristic score for move ordering
            """
            x, y = mv
            score = 0
            
            # 1. Proximity to last move (tactical focus)
            if last_positions:
                min_dist = min(abs(x - lx) + abs(y - ly) for lx, ly in last_positions)
                score += (10 - min_dist) * 100
            
            # 2. Center control bonus
            center = Defines.GRID_NUM // 2
            dist_to_center = abs(x - center) + abs(y - center)
            score += (20 - dist_to_center) * 10
            
            # 3. Quick pattern check (count neighbors)
            neighbors = 0
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM:
                    if board[nx][ny] != Defines.NOSTONE:
                        neighbors += 1
            score += neighbors * 50
        
            return score
        # Rank according to heuristic
        sorted_moves = sorted(
            list(candidates),
            key=move_score,
            reverse=True
        )
        
        return sorted_moves[:max_candidates]

    def pattern_evaluate(self, coords=None):
        """
        Evaluate board position based on pattern recognition.
        
        Parameters
        ----------
        coords : list of tuple, optional
            Stone coordinates (default: None)
            
        Returns
        -------
        float
            Pattern-based evaluation score
        """
        return self.fast_pattern_evaluate(self.np_board, Defines.BLACK, Defines.WHITE, Defines.NOSTONE, 6)
    
    def influence_evaluate(self,):
        """
        Evaluate board position based on influence/control.
        
        Returns
        -------
        float
            Influence-based evaluation score
        """
        board_np = self.np_board
        return self.fast_influence_evaluate(board_np, Defines.BLACK, Defines.WHITE, Defines.NOSTONE)
    
    def evaluate_board(self, board, last_positions):
        """
        Evaluate current board position using weighted heuristics.
        
        Combines influence and pattern evaluation with configurable weights.
        
        Parameters
        ----------
        board : list of list
            Current board state
        last_positions : tuple or StoneMove
            Last move played
            
        Returns
        -------
        float
            Overall evaluation score (positive favors BLACK, negative favors WHITE)
        """

        result = check_game_result_numpy(self.np_board, last_positions)
        if result == Defines.BLACK:  return Defines.MAXINT
        if result == Defines.WHITE:  return Defines.MININT
        if result == Defines.DRAW:   return 0

        coords = [(x, y)
                for x in range(Defines.GRID_NUM)
                for y in range(Defines.GRID_NUM)
                if board[x][y] != Defines.NOSTONE]

        if not coords:
            return 0

        influence_score = self.influence_evaluate()
        pattern_score = self.pattern_evaluate(board, coords)
        total = (
            self.weights["influence"] * influence_score
            + self.weights["pattern"] * pattern_score
        )

        # store in transposition table before returning
        # self.transposition_table[state_key] = total
        return total
    
    def quiescence_search(self, alpha, beta, maximizing_player, last_move, depth_limit=2):
        """
        Perform quiescence search to resolve tactical sequences.
        
        Extends search in tactical positions with immediate threats to avoid
        horizon effects and improve tactical awareness.
        
        Parameters
        ----------
        alpha : float
            Best score for maximizing player
        beta : float
            Best score for minimizing player
        maximizing_player : bool
            True if current player is maximizing
        last_move : tuple or StoneMove
            Last move played
        depth_limit : int, optional
            Maximum quiescence depth (default: 2)
            
        Returns
        -------
        float
            Quiescence evaluation score
        """
        board_np = self.np_board
        color = Defines.BLACK if maximizing_player else Defines.WHITE
        BLACK = Defines.BLACK
        WHITE = Defines.WHITE
        NOSTONE = Defines.NOSTONE
        self.metrics["quiescence_calls"]+=1

        return self.fast_quiescence_search(
            board_np, alpha, beta, maximizing_player, last_move, color, BLACK, WHITE, NOSTONE, depth_limit
    )
    
    def fast_quiescence_search(self, board_np, alpha, beta, maximizing_player, last_move, color, BLACK, WHITE, NOSTONE, depth_limit):
        """
        Fast quiescence search implementation.
        
        Parameters
        ----------
        board_np : numpy.ndarray
            Board as NumPy array
        alpha : float
            Best score for maximizing player
        beta : float
            Best score for minimizing player
        maximizing_player : bool
            True if current player is maximizing
        last_move : tuple
            Last move coordinate
        color : int
            Current player color
        BLACK : int
            Black stone constant
        WHITE : int
            White stone constant
        NOSTONE : int
            Empty cell constant
        depth_limit : int
            Remaining quiescence depth
            
        Returns
        -------
        float
            Quiescence score
        """
        stand_pat = self.influence_evaluate()
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat
        if depth_limit <= 0:
            return stand_pat
        threats = numpy_immediate_threats(board_np, color, BLACK, WHITE, NOSTONE)
        if len(threats) == 0:
            return stand_pat
        for k in range(len(threats)):
            x, y = threats[k]
            board_np[x, y] = color
            score = -self.fast_quiescence_search(board_np, -beta, -alpha, not maximizing_player, [(x, y)], 
                                            self.opponent_color(color), BLACK, WHITE, NOSTONE, depth_limit-1)
            board_np[x, y] = NOSTONE
            if maximizing_player:
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            else:
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
        return alpha if maximizing_player else beta
    
    def order_moves(self, board, candidates):
        """
        Order candidate moves using heuristics for better pruning.
        
        Prioritizes moves based on proximity to occupied stones, center control,
        and connectivity to improve alpha-beta pruning efficiency.
        
        Parameters
        ----------
        board : list of list
            Current board state
        candidates : list of StoneMove
            Candidate moves to order
            
        Returns
        -------
        list of StoneMove
            Sorted candidate moves (best first)
        """
        center = Defines.GRID_NUM // 2
        GRID = Defines.GRID_NUM

        # Create occupied set and NumPy versions for fast operations
        occupied = set(
            (x, y)
            for x in range(GRID)
            for y in range(GRID)
            if board[x][y] != Defines.NOSTONE
        )

        if not occupied:
            # Rare case: board is empty, nothing to sort
            return candidates

        occupied_arr = np.array(list(occupied), dtype=np.int32)
        board_np = np.array([[board[x][y] != Defines.NOSTONE for y in range(GRID)] for x in range(GRID)], dtype=bool)
        neighbor_offsets = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]

        def min_manhattan(pos, occarr):
            """
            Compute minimum Manhattan distance to any occupied cell.
            
            Parameters
            ----------
            pos : numpy.ndarray
                Position [x, y]
            occarr : numpy.ndarray
                Array of occupied positions (N x 2)
                
            Returns
            -------
            int
                Minimum Manhattan distance
            """
            # pos: np.array([x, y]), occarr: N x 2 array
            return np.min(np.abs(pos[0] - occarr[:,0]) + np.abs(pos[1] - occarr[:,1]))

        def neighbor_count(x, y):
            """
            Count neighboring stones around a position.
            
            Parameters
            ----------
            x : int
                Row coordinate
            y : int
                Column coordinate
                
            Returns
            -------
            int
                Number of neighboring stones
            """
            n = 0
            for dx, dy in neighbor_offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID and 0 <= ny < GRID:
                    if board_np[nx, ny]:
                        n += 1
            return n

        def move_priority(move):
            """
            Calculate priority score for move ordering.
            
            Parameters
            ----------
            move : StoneMove
                Candidate move to evaluate
                
            Returns
            -------
            float
                Priority score (higher is better)
            """
            score = 0
            pos1, pos2 = move.positions[0], move.positions[1]

            # Manhattan proximity to any occupied cell (vectorized)
            if occupied_arr.shape[0]:
                p1 = np.array([pos1.x, pos1.y])
                min_dist = min_manhattan(p1, occupied_arr)
                if pos2.x != 0:
                    p2 = np.array([pos2.x, pos2.y])
                    min_dist2 = min_manhattan(p2, occupied_arr)
                    min_dist = min(min_dist, min_dist2)
                score += (10 - min_dist) * 1000

            # Center control (as before)
            dist1 = abs(pos1.x - center) + abs(pos1.y - center)
            dist2 = abs(pos2.x - center) + abs(pos2.y - center) if pos2.x != 0 else 0
            score -= (dist1 + dist2) * 5

            # Fast NumPy-based neighbor count
            neighbors = 0
            if pos1.x != 0:
                neighbors += neighbor_count(pos1.x, pos1.y)
            if pos2.x != 0:
                neighbors += neighbor_count(pos2.x, pos2.y)
            score += neighbors * 50

            return score

        return sorted(candidates, key=move_priority, reverse=True)

    @staticmethod
    @numba.njit
    def fast_pattern_evaluate(board, BLACK, WHITE, NOSTONE, win_length=6):
        """
        Fast pattern-based evaluation using Numba JIT compilation.
        
        Detects and scores patterns of consecutive stones with varying
        numbers of open ends using vectorized operations.
        
        Parameters
        ----------
        board : numpy.ndarray
            Board state as 2D NumPy array
        BLACK : int
            Black stone constant
        WHITE : int
            White stone constant
        NOSTONE : int
            Empty cell constant
        win_length : int, optional
            Winning sequence length (default: 6)
            
        Returns
        -------
        float
            Pattern evaluation score (positive favors BLACK)
        """
        N = board.shape[0]
        # weights: index [stones][open_ends]
        weights = np.zeros((7, 3))
        weights[6, 0] = 100000
        weights[5, 2] = 50000
        weights[5, 1] = 20000
        weights[4, 2] = 8000
        weights[4, 1] = 3500
        weights[3, 2] = 800
        weights[3, 1] = 300
        weights[2, 2] = 80
        weights[2, 1] = 40

        directions = np.array([
            [0, 1], [1, 0], [1, 1], [1, -1]
        ])
        visited = np.zeros((N, N, 4), dtype=np.bool_)

        black_score = 0.0
        white_score = 0.0

        for x in range(N):
            for y in range(N):
                color = board[x, y]
                if color == NOSTONE:
                    continue
                for d in range(4):
                    if visited[x, y, d]:
                        continue
                    visited[x, y, d] = True

                    dx, dy = directions[d]
                    cnt, open_ends = 1, 0

                    # Forward
                    nx, ny = x + dx, y + dy
                    while 0 <= nx < N and 0 <= ny < N and board[nx, ny] == color:
                        cnt += 1
                        visited[nx, ny, d] = True
                        nx += dx
                        ny += dy
                    if 0 <= nx < N and 0 <= ny < N and board[nx, ny] == NOSTONE:
                        open_ends += 1

                    # Backward
                    nx, ny = x - dx, y - dy
                    while 0 <= nx < N and 0 <= ny < N and board[nx, ny] == color:
                        cnt += 1
                        visited[nx, ny, d] = True
                        nx -= dx
                        ny -= dy
                    if 0 <= nx < N and 0 <= ny < N and board[nx, ny] == NOSTONE:
                        open_ends += 1

                    if cnt > 6:
                        cnt = 6 # capping for lookup
                    v = weights[cnt, open_ends] if open_ends > 0 else weights[cnt, open_ends] * 0.5
                    if color == BLACK:
                        black_score += v
                    else:
                        white_score += v
        return black_score - white_score
    
    @staticmethod
    @numba.njit
    def fast_influence_evaluate(board, BLACK, WHITE, NOSTONE):
        """
        Fast influence-based evaluation using Numba JIT compilation.
        
        Evaluates board control by analyzing potential sequences in all
        directions with decay factors and blocking considerations.
        
        Parameters
        ----------
        board : numpy.ndarray
            Board state as 2D NumPy array
        BLACK : int
            Black stone constant
        WHITE : int
            White stone constant
        NOSTONE : int
            Empty cell constant
            
        Returns
        -------
        float
            Influence evaluation score (positive favors BLACK)
        """
        N = board.shape[0]
        # Weights can be precomputed as arrays
        weights = np.array([4096, 2048, 1024, 512, 256], dtype=np.float64)
        directions = np.array([
            [1, 0], [0, 1], [1, 1], [1, -1]
        ])
        direction_multipliers = np.array([1.5, 1.0, 1.0, 1.0])
        total_score = 0.0

        for x in range(N):
            for y in range(N):
                color = board[x, y]
                if color == NOSTONE:
                    continue
                color_sign = 1 if color == BLACK else -1
                for d in range(4):
                    dx, dy = directions[d]
                    dir_mult = direction_multipliers[d]
                    # Build the line
                    line = []
                    for step in range(-5, 6):
                        nx, ny = x + step*dx, y + step*dy
                        if 0 <= nx < N and 0 <= ny < N:
                            line.append(board[nx, ny])
                        else:
                            line.append(-99)  # Off-board indicator
                    line = np.array(line)
                    # Sliding window
                    for i in range(6):
                        window = line[i:i+6]
                        # Skip if any are off-board
                        if np.any(window == -99):
                            continue
                        same = np.sum(window == color)
                        empty = np.sum(window == NOSTONE)
                        opp = 6 - same - empty
                        if same <= 0:
                            continue
                        # Decay weighting
                        decay_factor = 0.0
                        for j in range(6):
                            if window[j] == color:
                                dist = abs(j - 3)
                                decay_factor += (0.95 ** dist)
                        decay_factor /= same
                        block_ratio = (6 - opp) / 6.0 if opp > 0 else 1.0
                        if same >= 6:
                            contrib = 1000000.0
                        else:
                            idx = min(same-1, 4)
                            contrib = weights[idx] * (2**empty)
                        contrib *= dir_mult * decay_factor * block_ratio * color_sign
                        total_score += contrib
        return total_score
    
    @staticmethod
    def opponent_color(color):
        """
        Get the opponent's stone color.
        
        Parameters
        ----------
        color : int
            Current player color (BLACK or WHITE)
            
        Returns
        -------
        int
            Opponent color (WHITE if BLACK, BLACK if WHITE, or NOSTONE)
        """
        if color == Defines.BLACK:
            return Defines.WHITE
        elif color == Defines.WHITE:
            return Defines.BLACK
        return Defines.NOSTONE
    
@numba.njit    
def numpy_immediate_threats(board_np, color, BLACK, WHITE, NOSTONE):
    """
    Find all immediate winning or blocking threats on the board.
    
    Identifies positions where placing a stone would immediately win for
    either the current player or opponent, enabling tactical responses.
    
    Parameters
    ----------
    board_np : numpy.ndarray
        Board state as 2D NumPy array
    color : int
        Current player color
    BLACK : int
        Black stone constant
    WHITE : int
        White stone constant
    NOSTONE : int
        Empty cell constant
        
    Returns
    -------
    numpy.ndarray
        Array of (x, y) threat positions, shape (N, 2)
    """
    N = board_np.shape[0]

    opponent = BLACK if color == WHITE else WHITE
    MAX_THREATS = 1000  # An upper bound that's safe for your board size
    threats = np.zeros((MAX_THREATS,2), dtype=np.int32)
    num_threats = 0
    # Find all NOSTONE positions
    empties = np.argwhere(board_np == NOSTONE)
    for i in range(empties.shape[0]):
        x, y = empties[i]
        # Try as current color
        board_np[x, y] = color
        win = check_game_result_numpy(board_np, x, y, color)
        board_np[x, y] = NOSTONE
        if win:
            threats[num_threats,0] = x
            threats[num_threats,1] = y
            num_threats += 1
            continue
        # Try as opponent
        board_np[x, y] = opponent
        opp_win = check_game_result_numpy(board_np, x, y, opponent)
        board_np[x, y] = NOSTONE
        if opp_win:
            threats[num_threats,0] = x
            threats[num_threats,1] = y
            num_threats += 1
            continue
    return threats[:num_threats]

@numba.njit
def check_game_result_numpy(board, x, y, color, win_length=6):
    """
    Check if a move at (x, y) creates a winning sequence.
    
    Examines all four directions (horizontal, vertical, and two diagonals)
    to detect win_length consecutive stones of the same color.
    
    Parameters
    ----------
    board : numpy.ndarray
        Board state as 2D NumPy array
    x : int
        Row coordinate of move
    y : int
        Column coordinate of move
    color : int
        Stone color to check
    win_length : int, optional
        Number of consecutive stones needed to win (default: 6)
        
    Returns
    -------
    bool
        True if the move creates a winning sequence, False otherwise
    """
    N = board.shape[0]
    directions = np.array([
        (1, 0),   # vertical
        (0, 1),   # horizontal
        (1, 1),   # main diagonal
        (1, -1),  # anti-diagonal
    ])
    for dx, dy in directions:
        count = 1
        # Forward direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < N and 0 <= ny < N and board[nx, ny] == color:
            count += 1
            nx += dx
            ny += dy
        # Backward direction
        nx, ny = x - dx, y - dy
        while 0 <= nx < N and 0 <= ny < N and board[nx, ny] == color:
            count += 1
            nx -= dx
            ny -= dy
        if count >= win_length:
            return True
    return False

    
def flush_output():
    import sys
    sys.stdout.flush()
