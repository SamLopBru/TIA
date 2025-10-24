from defines import *
import time
from itertools import combinations



DIRECTIONS_MAIN = [(1, 0), (0, 1), (1, 1), (1, -1)]
DIRECTIONS_ADJ  = [(dx, dy) for dx in (-1, 0, 1)
                             for dy in (-1, 0, 1)
                             if not (dx == 0 and dy == 0)]

# Point (x, y) if in the valid position of the board.
def isValidPos(x,y):
    return x>0 and x<Defines.GRID_NUM-1 and y>0 and y<Defines.GRID_NUM-1
    
def init_board(board):
    for i in range(21):
        board[i][0] = board[0][i] = board[i][Defines.GRID_NUM - 1] = board[Defines.GRID_NUM - 1][i] = Defines.BORDER
    for i in range(1, Defines.GRID_NUM - 1):
        for j in range(1, Defines.GRID_NUM - 1):
            board[i][j] = Defines.NOSTONE
          
def make_move(board, move, color):
    board[move.positions[0].x][move.positions[0].y] = color
    board[move.positions[1].x][move.positions[1].y] = color

def unmake_move(board, move):
    board[move.positions[0].x][move.positions[0].y] = Defines.NOSTONE
    board[move.positions[1].x][move.positions[1].y] = Defines.NOSTONE

def is_win_by_premove(board, preMove):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for direction in directions:
        for i in range(len(preMove.positions)):
            count = 0
            position = preMove.positions[i]
            n = x = position.x
            m = y = position.y
            movStone = board[n][m]
            
            if (movStone == Defines.BORDER or movStone == Defines.NOSTONE):
                return False;
                
            while board[x][y] == movStone:
                x += direction[0]
                y += direction[1]
                count += 1
            x = n - direction[0]
            y = m - direction[1]
            while board[x][y] == movStone:
                x -= direction[0]
                y -= direction[1]
                count += 1
            if count >= 6:
                return True
    return False

def get_msg(max_len):
    buf = input().strip()
    return buf[:max_len]

def log_to_file(msg):
    g_log_file_name = Defines.LOG_FILE
    try:
        with open(g_log_file_name, "a") as file:
            tm = time.time()
            ptr = time.ctime(tm)
            ptr = ptr[:-1]
            file.write(f"[{ptr}] - {msg}\n")
        return 0
    except Exception as e:
        print(f"Error: Can't open log file - {g_log_file_name}")
        return -1

def move2msg(move):
    if move.positions[0].x == move.positions[1].x and move.positions[0].y == move.positions[1].y:
        msg = f"{chr(ord('S') - move.positions[0].x + 1)}{chr(move.positions[0].y + ord('A') - 1)}"
        return msg
    else:
        msg = f"{chr(move.positions[0].y + ord('A') - 1)}{chr(ord('S') - move.positions[0].x + 1)}" \
              f"{chr(move.positions[1].y + ord('A') - 1)}{chr(ord('S') - move.positions[1].x + 1)}"
        return msg

def msg2move(msg):
    move = StoneMove()
    if len(msg) == 2:
        move.positions[0].x = move.positions[1].x = ord('S') - ord(msg[1]) + 1
        move.positions[0].y = move.positions[1].y = ord(msg[0]) - ord('A') + 1
        move.score = 0
        return move
    else:
        move.positions[0].x = ord('S') - ord(msg[1]) + 1
        move.positions[0].y = ord(msg[0]) - ord('A') + 1
        move.positions[1].x = ord('S') - ord(msg[3]) + 1
        move.positions[1].y = ord(msg[2]) - ord('A') + 1
        move.score = 0
        return move

def print_board(board: list[list[int]], preMove=None):
    print("   " + "".join([chr(i + ord('A') - 1)+" " for i in range(1, Defines.GRID_NUM - 1)]))
    for i in range(1, Defines.GRID_NUM - 1):
        print(f"{chr(ord('A') - 1 + i)}", end=" ")
        for j in range(1, Defines.GRID_NUM - 1):
            x = Defines.GRID_NUM - 1 - j
            y = i
            stone = board[x][y]
            if stone == Defines.NOSTONE:
                print(" -", end="")
            elif stone == Defines.BLACK:
                print(" O", end="")
            elif stone == Defines.WHITE:
                print(" *", end="")
        print(" ", end="")        
        print(f"{chr(ord('A') - 1 + i)}", end="\n")
    print("   " + "".join([chr(i + ord('A') - 1)+" " for i in range(1, Defines.GRID_NUM - 1)]))

def print_score(move_list:list, n):
    board = [[0] * Defines.GRID_NUM for _ in range(Defines.GRID_NUM)]
    for move in move_list:
        board[move.x][move.y] = move.score

    print("  " + "".join([f"{i:4}" for i in range(1, Defines.GRID_NUM - 1)]))
    for i in range(1, Defines.GRID_NUM - 1):
        print(f"{i:2}", end="")
        for j in range(1, Defines.GRID_NUM - 1):
            score = board[i][j]
            if score == 0:
                print("   -", end="")
            else:
                print(f"{score:4}", end="")
        print()

def check_game_result(board, last_positions, engine=None):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    if last_positions is None: 
        return Defines.NOSTONE

    # Normalize last_positions -> list of (x,y)
    norm_positions = []
    if hasattr(last_positions, "positions"):  # StoneMove
        for p in last_positions.positions:
            norm_positions.append((p.x, p.y))
    elif isinstance(last_positions, (list, tuple)):
        for pos in last_positions:
            if hasattr(pos, "x"):  # StonePosition
                norm_positions.append((pos.x, pos.y))
            else:
                norm_positions.append(pos)
    else:
        norm_positions.append((last_positions.x, last_positions.y))

    # --- Check win ---
    for (n, m) in norm_positions:
        movStone = board[n][m]
        if movStone == Defines.NOSTONE or movStone == Defines.BORDER:
            continue

        for dr, dc in directions:
            count = 1
            # forward
            x, y = n + dr, m + dc
            while 0 <= x < Defines.GRID_NUM and 0 <= y < Defines.GRID_NUM and board[x][y] == movStone:
                count += 1; x += dr; y += dc
            # backward
            x, y = n - dr, m - dc
            while 0 <= x < Defines.GRID_NUM and 0 <= y < Defines.GRID_NUM and board[x][y] == movStone:
                count += 1; x -= dr; y -= dc
            if count >= 6:
                return movStone

    # --- Check draw using counter ---
    total_cells = Defines.GRID_NUM * Defines.GRID_NUM
    if engine and engine.stone_count == total_cells:
        return Defines.DRAW

    return Defines.NOSTONE


def move_heuristic(board, mv, last_positions=None, my_color=None) -> int:
    x, y = mv
    if board[x][y] != Defines.NOSTONE:
        return -9999  # invalid move

    GRID_NUM = Defines.GRID_NUM
    score = 0

    opp_color = Defines.WHITE if my_color == Defines.BLACK else Defines.BLACK
    get = board.__getitem__  # local alias for slightly faster access

    # --- (1) Tactical lines: offense + defense ---
    for dx, dy in DIRECTIONS_MAIN:
        my_count = opp_count = my_open = opp_open = 0

        # ---- Scan my stones in both directions ----
        for sign in (1, -1):
            nx, ny = x + sign * dx, y + sign * dy
            while 0 <= nx < GRID_NUM and 0 <= ny < GRID_NUM:
                val = get(nx)[ny]
                if val == my_color:
                    my_count += 1
                elif val == Defines.NOSTONE:
                    my_open += 1
                    break
                else:
                    break
                nx += sign * dx
                ny += sign * dy

        # ---- Scan opponent stones ----
        for sign in (1, -1):
            nx, ny = x + sign * dx, y + sign * dy
            while 0 <= nx < GRID_NUM and 0 <= ny < GRID_NUM:
                val = get(nx)[ny]
                if val == opp_color:
                    opp_count += 1
                elif val == Defines.NOSTONE:
                    opp_open += 1
                    break
                else:
                    break
                nx += sign * dx
                ny += sign * dy

        # --- Offensive scoring ---
        if my_count >= 5:
            score += 15000
        elif my_count == 4:
            score += 4000 * my_open
        elif my_count == 3:
            score += 700 * my_open
        elif my_count == 2:
            score += 120 * my_open
        elif my_count == 1:
            score += 30

        # --- Defensive scoring ---
        if opp_count >= 5:
            score += 12000
        elif opp_count == 4:
            score += 3500 * opp_open
        elif opp_count == 3:
            score += 600 * opp_open
        elif opp_count == 2:
            score += 100 * opp_open

    # --- (2) Local adjacency density ---
    for dx, dy in DIRECTIONS_ADJ:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_NUM and 0 <= ny < GRID_NUM:
            val = get(nx)[ny]
            if val == my_color:
                score += 20
            elif val == opp_color:
                score += 10

    # --- (3) Proximity to last moves ---
    if last_positions:
        for (lx, ly) in last_positions:
            dist = abs(x - lx) + abs(y - ly)
            if dist <= 6:
                score += 16 - 2 * dist  # linear falloff

    # --- (4) Center preference ---
    c = GRID_NUM // 2
    distc = abs(x - c) + abs(y - c)
    score += max(0, 6 - distc) * 2

    return score

def generate_candidate_pairs(candidates, max_pairs=30):
    pairs = []
    seen = set()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a == b: 
                continue
            # skip if already seen
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)

            move = StoneMove()
            move.positions[0] = StonePosition(*a)
            move.positions[1] = StonePosition(*b)
            pairs.append(move)
            if len(pairs) >= max_pairs:
                return pairs
    return pairs