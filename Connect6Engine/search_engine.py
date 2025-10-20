from tools import *
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

    def before_search(self, board, color, alphabeta_depth):
        self.m_board = [row[:] for row in board]
        # count stones already on board
        self.stone_count = sum(1 for r in self.m_board for v in r if v != Defines.NOSTONE)
        self.m_chess_type = color
        self.m_alphabeta_depth = alphabeta_depth
        self.m_total_nodes = 0

    def board_hash(self, board):
        """Compute a simple but unique hash for board state."""
        import hashlib, json
        flat = ''.join(str(cell) for row in board for cell in row)
        return hashlib.md5(flat.encode()).hexdigest()

    def alpha_beta_pruning(self, board, depth, alpha, beta, maximizing_player, last_move, max_candidates=15):
        """
        Args:
        board: 2D board (GRID_NUM x GRID_NUM)
        depth: current remaining search depth
        alpha: best value that MAX has guaranteed
        beta: best value that MIN has guaranteed
        maximizing_player: True if Black's turn (MAX), False if White's turn (MIN)
        last_move: the last move(s) played (StoneMove or Position object(s))
        max_candidates: max number of candidates considered for branching

        Returns:
            (score, move) -> numeric evaluation and best move found at this node
        """
        # terminal conditions
        result = check_game_result(board, last_move)
        if result == Defines.BLACK:
            return (Defines.MAXINT, None)
        elif result == Defines.WHITE:
            return (Defines.MININT, None)
        elif result == Defines.DRAW:
            return (0, None)

        if depth == 0:
            # extend tactical lines for stability
            q_score = self.quiescence_search(board, alpha, beta, maximizing_player, last_move)
            return (q_score, None)

        
        threats = self.immediate_threats(board, Defines.BLACK if maximizing_player else Defines.WHITE)
        if len(threats) > 1:
            # create pseudo-moves (StoneMove objects) from threats and return early
            best_threat, second_threat = threats[0], threats[1]
            best_move = StoneMove()
            best_move.positions = [StonePosition(best_threat[0], best_threat[1]),StonePosition(second_threat[0], second_threat[1])]

            return (Defines.MAXINT // 2, best_move)

        # generate StoneMove candidates
        singles = self.generate_candidate_moves(board, last_move, max_candidates)
        candidates = generate_candidate_pairs(singles, max_pairs=10)  # returns StoneMoves

        if not candidates:
            return (self.evaluate_board(board, last_move), None)

        best_move = None
        value = -float("inf") if maximizing_player else float("inf")

        for move in candidates:
            pos1 = move.positions[0]
            pos2 = move.positions[1]

            color = Defines.BLACK if maximizing_player else Defines.WHITE

            # Apply move
            board[pos1.x][pos1.y] = color
            if not (pos2.x == 0 and pos2.y == 0):  # skip dummy second move for first turn
                board[pos2.x][pos2.y] = color

            # Recursive call
            eval_score, _ = self.alpha_beta_pruning(
                board, depth - 1, alpha, beta, not maximizing_player, move, max_candidates
            )

            # Undo move
            board[pos1.x][pos1.y] = Defines.NOSTONE
            if not (pos2.x == 0 and pos2.y == 0):
                board[pos2.x][pos2.y] = Defines.NOSTONE

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

            # pruning
            if beta <= alpha:
                break

        return value, best_move
        
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

        # If the board is empty → play in the center
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

        # Rank according to heuristic
        sorted_moves = sorted(
            list(candidates),
            key=lambda mv: move_heuristic(board, mv, last_positions),
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
        state_key = self.board_hash(board)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

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
        self.transposition_table[state_key] = total
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
    
def flush_output():
    import sys
    sys.stdout.flush()
