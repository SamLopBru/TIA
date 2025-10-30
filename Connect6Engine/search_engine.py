from tools import *
import random
# 'influence': 0.4833960258781689, 'pattern': 0.5166039741218311

class SearchEngine():
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
        self.transposition_table = {}

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
        """Reset all metrics before a new search"""
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
        Each cell (x,y) and each possible stone type (BLACK, WHITE)
        gets a random 64-bit integer.
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
        Compute Zobrist hash from scratch (only when needed).
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
        Incrementally update hash when placing/removing a stone.
        Calling this twice with same (x,y,color) restores original hash.
        """
        if color not in (Defines.BLACK, Defines.WHITE):
            return
        self.current_hash ^= self.zobrist_table[(x, y, color)]

    def alpha_beta_pruning(self, board, depth, alpha, beta, maximizing_player, last_move, max_candidates=40, is_root=False):
        
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
            q_score = self.quiescence_search(board, alpha, beta, maximizing_player, last_move)
            return (q_score, None)

        # Check immediate threats
        threats = self.immediate_threats(board, Defines.BLACK if maximizing_player else Defines.WHITE)
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
        
        self.transposition_table[state_key] = (depth, value, best_move, flag)

        return value, best_move  # ✅ ALWAYS 2 values
    
    def check_first_move(self):
        for i in range(1,len(self.m_board)-1):
            for j in range(1, len(self.m_board[i])-1):
                if(self.m_board[i][j] != Defines.NOSTONE):
                    return False
        return True

    def generate_candidate_moves(self, board, last_move=None, max_candidates=15, radius=3):
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

    def pattern_evaluate(self, board, coords):
        """
        Tactical pattern-based evaluation that accounts equally for vertical,
        horizontal, and both diagonal directions.
        """
        black_score = 0
        white_score = 0

        # Use correct direction mapping (row, col)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # Pattern weights for count/open_ends
        pattern_weights = {
            (6, 0): 100000,
            (5, 2): 50000, (5, 1): 20000,
            (4, 2): 8000,  (4, 1): 3500,
            (3, 2): 800,   (3, 1): 300,
            (2, 2): 80,    (2, 1): 40,
        }

        checked = set()  # avoid recounting same lines

        for (x, y) in coords:
            color = board[x][y]
            if color == Defines.NOSTONE:
                continue

            for dx, dy in directions:
                key = (x, y, dx, dy)
                if key in checked:
                    continue
                checked.add(key)

                count, open_ends = 1, 0

                # Forward direction
                nx, ny = x + dx, y + dy
                while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == color:
                    count += 1
                    checked.add((nx, ny, dx, dy))
                    nx += dx
                    ny += dy
                if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == Defines.NOSTONE:
                    open_ends += 1

                # Backward direction
                nx, ny = x - dx, y - dy
                while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == color:
                    count += 1
                    checked.add((nx, ny, dx, dy))
                    nx -= dx
                    ny -= dy
                if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == Defines.NOSTONE:
                    open_ends += 1

                # Apply pattern value (blocked lines get low weight)
                base_val = pattern_weights.get((count, open_ends), 0)
                if open_ends == 0 and count < 6:
                    base_val *= 0.5  # blocked pattern penalty

                if color == Defines.BLACK:
                    black_score += base_val
                else:
                    white_score += base_val

        return black_score - white_score
    
    def influence_evaluate(self, board):
        """
        Enhanced influence evaluation:
        - Uses geometric decay to reduce long-distance influence
        - Applies direction multipliers for balancing vertical/horizontal/diagonal importance
        - Smooth handling of mixed-color windows (partial blocking)
        - Computes weighted contribution per color
        """
        GRID = Defines.GRID_NUM
        weights = [2**12, 2**11, 2**10, 2**9, 2**8]  # "same" stones 1..5
        empty_weight = 2

        # Four canonical directions (vertical, horizontal, diagonals)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        # You can tweak these multipliers if one direction tends to dominate or lag.
        direction_multipliers = {
            (1, 0): 1.5,   # vertical
            (0, 1): 1.0,   # horizontal
            (1, 1): 1.0,   # main diagonal
            (1, -1): 1.0,  # anti-diagonal
        }

        total_score = 0

        for x in range(GRID):
            for y in range(GRID):
                color = board[x][y]
                if color == Defines.NOSTONE:
                    continue

                color_sign = 1 if color == Defines.BLACK else -1

                for dx, dy in directions:
                    dir_mult = direction_multipliers[(dx, dy)]

                    # Collect up to 11 cells (5 before, current, 5 after)
                    line = []
                    for step in range(-5, 6):
                        nx, ny = x + step * dx, y + step * dy
                        if 0 <= nx < GRID and 0 <= ny < GRID:
                            line.append(board[nx][ny])
                        else:
                            line.append(None)  # outside bounds

                    # Scan sliding window of 6 cells
                    for i in range(len(line) - 5):
                        window = line[i:i + 6]

                        # Count values
                        same = window.count(color)
                        empty = window.count(Defines.NOSTONE)
                        opp = 6 - same - empty if None not in window else 0  # within bounds

                        # skip all-empty or off-board windows
                        if same <= 0 or None in window:
                            continue

                        # Compute distance-based geometric decay weighting:
                        # closer stones have higher influence
                        mid = i + 3
                        decay_factor = 0.0
                        for j, cell in enumerate(window):
                            if cell == color:
                                dist = abs(j - 3)  # distance from center
                                decay_factor += (0.95 ** dist)
                        decay_factor /= same  # normalize per same-colored stones

                        # Smoothed mixed-color handling:
                        # If mixed (both colors in window), scale down proportionally
                        if opp > 0:
                            block_ratio = (6 - opp) / 6.0  # e.g. 1.0 if no opponent, lower otherwise
                        else:
                            block_ratio = 1.0

                        # Determine index safely
                        if same >= 6:
                            contrib = 1000000.0
                        else:
                            idx = min(same - 1, len(weights) - 1)
                            contrib = weights[idx] * (empty_weight ** empty)

                        # Apply all modifiers and accumulate
                        contrib *= dir_mult * decay_factor * block_ratio * color_sign
                        total_score += contrib

        return total_score

    def evaluate_board(self, board, last_positions):
        # state_key = hex(self.compute_board_hash(board))
        # if state_key in self.transposition_table:
        #     return self.transposition_table[state_key]

        result = check_game_result(board, last_positions)
        if result == Defines.BLACK:  return Defines.MAXINT
        if result == Defines.WHITE:  return Defines.MININT
        if result == Defines.DRAW:   return 0

        coords = [(x, y)
                for x in range(Defines.GRID_NUM)
                for y in range(Defines.GRID_NUM)
                if board[x][y] != Defines.NOSTONE]

        if not coords:
            return 0

        influence_score = self.influence_evaluate(board)
        pattern_score = self.pattern_evaluate(board, coords)
        total = (
            self.weights["influence"] * influence_score
            + self.weights["pattern"] * pattern_score
        )

        # store in transposition table before returning
        # self.transposition_table[state_key] = total
        return total

    def immediate_threats(self, board, color):
        """
        Detect immediate winning or blocking threats.
        Returns a list of (x, y) positions that either
        complete 6-in-a-row or block the opponent's imminent win.
        """
        threats = []
        opponent = Defines.BLACK if color == Defines.WHITE else Defines.WHITE

        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                if board[x][y] != Defines.NOSTONE:
                    continue

                # --- Simulate placing a stone for the current player ---
                board[x][y] = color
                if check_game_result(board, [(x, y)]) == color:
                    # this move wins immediately
                    threats.append((x, y))
                    board[x][y] = Defines.NOSTONE
                    continue
                board[x][y] = Defines.NOSTONE

                # --- Simulate opponent placing a stone (check defense) ---
                board[x][y] = opponent
                if check_game_result(board, [(x, y)]) == opponent:
                    # opponent could win here next move → must block
                    threats.append((x, y))
                board[x][y] = Defines.NOSTONE

        return threats

    def quiescence_search(self, board, alpha, beta, maximizing_player, last_move, depth_limit=2):
        """Extend search along noisy tactical lines (e.g., immediate threats)."""
        self.metrics['quiescence_calls'] += 1
        
        stand_pat = self.evaluate_board(board, last_move)
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

        threats = self.immediate_threats(board, Defines.BLACK if maximizing_player else Defines.WHITE)
        if not threats:
            return stand_pat

        for (x, y) in threats:
            color = Defines.BLACK if maximizing_player else Defines.WHITE
            board[x][y] = color
            score = -self.quiescence_search(board, -beta, -alpha, not maximizing_player, [(x, y)], depth_limit-1)
            board[x][y] = Defines.NOSTONE

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
        center = Defines.GRID_NUM // 2
        
        # Pre-compute occupied positions ONCE
        occupied = set()
        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                if board[x][y] != Defines.NOSTONE:
                    occupied.add((x, y))
        
        def move_priority(move):
            score = 0
            pos1, pos2 = move.positions[0], move.positions[1]
            
            # 1. Proximity to ANY occupied square (precomputed)
            if occupied:
                min_dist = min(
                    abs(pos1.x - ox) + abs(pos1.y - oy) 
                    for ox, oy in occupied
                )
                if pos2.x != 0:
                    min_dist2 = min(
                        abs(pos2.x - ox) + abs(pos2.y - oy) 
                        for ox, oy in occupied
                    )
                    min_dist = min(min_dist, min_dist2)
                score += (10 - min_dist) * 1000
            
            # 2. Center control
            dist1 = abs(pos1.x - center) + abs(pos1.y - center)
            dist2 = abs(pos2.x - center) + abs(pos2.y - center) if pos2.x != 0 else 0
            score -= (dist1 + dist2) * 5
            
            # 3. Neighbor count (quick lookup from precomputed set)
            neighbors = 0
            for pos in [pos1, pos2]:
                if pos.x == 0:
                    continue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        if (pos.x + dx, pos.y + dy) in occupied:
                            neighbors += 1
            score += neighbors * 50
            
            return score
        
        return sorted(candidates, key=move_priority, reverse=True)
        
    

def flush_output():
    import sys
    sys.stdout.flush()
