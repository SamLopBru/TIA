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

        # --- heuristic evaluation ---
        black_score = 0
        white_score = 0
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        pattern_weights = {
            (6, 0): 100000,   # win
            (5, 2): 50000,    # open five
            (5, 1): 20000,    # semi-open five
            (4, 2): 10000,
            (4, 1): 5000,
            (3, 2): 1000,
            (3, 1): 500,
            (2, 2): 50,
            (2, 1): 25,
        }

        for (x, y) in coords:
            color = board[x][y]
            if color == Defines.NOSTONE:
                continue

            for dx, dy in directions:
                count, open_ends = 1, 0

                # forward
                nx, ny = x + dx, y + dy
                while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == color:
                    count += 1
                    nx += dx
                    ny += dy
                if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == Defines.NOSTONE:
                    open_ends += 1

                # backward
                nx, ny = x - dx, y - dy
                while 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == color:
                    count += 1
                    nx -= dx
                    ny -= dy
                if 0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and board[nx][ny] == Defines.NOSTONE:
                    open_ends += 1

                score_val = pattern_weights.get((count, open_ends), 0)
                if color == Defines.BLACK:
                    black_score += score_val
                else:
                    white_score += score_val

        return black_score - white_score
    
    def minimax(self, board, depth, maximizing_player, last_move, max_candidates=10): #not in use
        """
        Naive Minimax search without pruning.

        Args:
            board: 2D board (GRID_NUM x GRID_NUM)
            depth: remaining search depth to explore
            maximizing_player: True if Black's turn (MAX), False if White's turn (MIN)
            last_move: the last move(s) played (needed for win/draw checks)
            max_candidates: limit candidate positions to avoid explosion

        Returns:
            (score, move) -> numeric evaluation and the selected move(s)
        """

        # --- Check for terminal cases or end of depth ---
        result = check_game_result(board, last_move)
        if result == Defines.BLACK:
            return (Defines.MAXINT, None)
        elif result == Defines.WHITE:
            return (Defines.MININT, None)
        elif result == Defines.DRAW:
            return (0, None)

        if depth == 0:
            return (self.evaluate_board(board, last_move), None)

        # --- Candidate move generation ---
        candidates = self.generate_naive_moves(board, max_candidates)

        if len(candidates) < 2:
            return (self.evaluate_board(board, last_move), None)

        best_move = None

        if maximizing_player:  # Black tries to maximize score
            max_eval = -float("inf")

            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    move1, move2 = candidates[i], candidates[j]

                    # Apply moves
                    board[move1[0]][move1[1]] = Defines.BLACK
                    board[move2[0]][move2[1]] = Defines.BLACK

                    eval_score, _ = self.minimax(
                        board, 
                        depth - 1, 
                        False, 
                        [StoneMove(move1[0], move1[1]), StoneMove(move2[0], move2[1])]
                    )

                    # Undo moves
                    board[move1[0]][move1[1]] = Defines.NOSTONE
                    board[move2[0]][move2[1]] = Defines.NOSTONE

                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = (move1, move2)

            return (max_eval, best_move)

        else:  # White tries to minimize score
            min_eval = float("inf")

            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    move1, move2 = candidates[i], candidates[j]

                    board[move1[0]][move1[1]] = Defines.WHITE
                    board[move2[0]][move2[1]] = Defines.WHITE

                    eval_score, _ = minimax(
                        board, 
                        depth - 1, 
                        True, 
                        [StoneMove(move1[0], move1[1]), StoneMove(move2[0], move2[1])]
                    )

                    board[move1[0]][move1[1]] = Defines.NOSTONE
                    board[move2[0]][move2[1]] = Defines.NOSTONE

                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = (move1, move2)

            return (min_eval, best_move)

def flush_output():
    import sys
    sys.stdout.flush()
