from tools import *

class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_alphabeta_depth = None
        self.m_total_nodes = 0
        self.last_positions = None
        self.stone_count = 0

    def before_search(self, board, color, alphabeta_depth):
        self.m_board = [row[:] for row in board]
        # count stones already on board
        self.stone_count = sum(1 for r in self.m_board for v in r if v != Defines.NOSTONE)
        self.m_chess_type = color
        self.m_alphabeta_depth = alphabeta_depth
        self.m_total_nodes = 0

    def alpha_beta_pruning(self, board, depth, alpha, beta, maximizing_player, last_move, max_candidates=15):
        """Args:
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
        # terminal conditions...
        result = check_game_result(board, last_move)
        if result == Defines.BLACK:
            return (Defines.MAXINT, None)
        elif result == Defines.WHITE:
            return (Defines.MININT, None)
        elif result == Defines.DRAW:
            return (0, None)

        if depth == 0:
            return (self.evaluate_board(board, last_move), None)

        # inside alpha_beta_pruning, before generate_candidate_moves
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
        """Tactical pattern-based evaluation (existing logic extracted)."""
        black_score = 0
        white_score = 0
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        pattern_weights = {
            (6, 0): 100000, (5, 2): 50000, (5, 1): 20000,
            (4, 2): 10000, (4, 1): 5000, (3, 2): 1000,
            (3, 1): 500, (2, 2): 50, (2, 1): 25,
        }

    
        for (x, y) in coords:
            color = board[x][y]
            if color == Defines.NOSTONE:
                continue

            for dx, dy in directions:
                count, open_ends = 1, 0

                nx, ny = x + dx, y + dy
                while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == color:
                    count += 1
                    nx += dx
                    ny += dy
                if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == Defines.NOSTONE:
                    open_ends += 1

                nx, ny = x - dx, y - dy
                while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == color:
                    count += 1
                    nx -= dx
                    ny -= dy
                if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == Defines.NOSTONE:
                    open_ends += 1

                val = pattern_weights.get((count, open_ends), 0)
                if color == Defines.BLACK:
                    black_score += val
                else:
                    white_score += val

        return black_score - white_score
    
    def influence_evaluate(self, board):
        weights = [2**12, 2**11, 2**10, 2**9, 2**8]  # same = 1..5
        empty_weight = 2
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total_score = 0

        for x in range(Defines.GRID_NUM):
            for y in range(Defines.GRID_NUM):
                color = board[x][y]
                if color == Defines.NOSTONE:
                    continue

                color_sign = 1 if color == Defines.BLACK else -1

                for dx, dy in directions:
                    line = []
                    # Collect up to 11 cells (5 before, current, 5 after)
                    for step in range(-5, 6):
                        nx, ny = x + step * dx, y + step * dy
                        if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM:
                            line.append(board[nx][ny])

                    # Slide a 6-cell window along the 11-cell line
                    for i in range(len(line) - 5):
                        window = line[i:i+6]

                        # Skip if blocked (both colors in window)
                        if Defines.BLACK in window and Defines.WHITE in window:
                            continue

                        same = window.count(color)
                        empty = window.count(Defines.NOSTONE)

                        # Guard against invalid indices
                        if same == 0:
                            continue
                        elif same >= 6:
                            # completed 6-in-a-row → strong win bonus
                            total_score += color_sign * 1000000
                        else:
                            total_score += color_sign * weights[same - 1] * (empty_weight ** empty)

        return total_score

    def evaluate_board(self, board: list[list[int]], last_positions):
        """
        Naive static evaluation function for Connect6. Only evaluates the lines 
        passing through the latest stones
        """

        # terminal states
        result = check_game_result(board, last_positions)
        if result == Defines.BLACK:  return Defines.MAXINT
        if result == Defines.WHITE:  return Defines.MININT
        if result == Defines.DRAW:   return 0

        # --- normalize last_positions to a list of (x,y) ---
        coords = []

        if last_positions is None:
            return 0

        if isinstance(last_positions, StoneMove):
            # add valid stones inside move
            for pos in last_positions.positions:
                if 0 <= pos.x < Defines.GRID_NUM and 0 <= pos.y < Defines.GRID_NUM:
                    if board[pos.x][pos.y] != Defines.NOSTONE:
                        coords.append((pos.x, pos.y))

        elif isinstance(last_positions, StonePosition):
            coords.append((last_positions.x, last_positions.y))

        elif isinstance(last_positions, (list, tuple)):
            # could be a list of StoneMoves, StonePositions, or tuples
            for p in last_positions:
                if isinstance(p, StoneMove):
                    for pos in p.positions:
                        if board[pos.x][pos.y] != Defines.NOSTONE:
                            coords.append((pos.x, pos.y))
                elif isinstance(p, StonePosition):
                    coords.append((p.x, p.y))
                elif isinstance(p, (list, tuple)):
                    if len(p) == 2:
                        coords.append((p[0], p[1]))

        else:
            print("⚠️ Unexpected type for last_positions:", type(last_positions))
            return 0

        influence_score = self.influence_evaluate(board)
        pattern_score = self.pattern_evaluate(board, coords)
        return 0.7 * influence_score + 0.3 * pattern_score

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

def flush_output():
    import sys
    sys.stdout.flush()
