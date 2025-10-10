# shared_board.py
import sys
sys.path.append('../Connect6Engine')
from defines import *
import threading

class SharedBoard:
    def __init__(self):
        self.board = [[Defines.NOSTONE for _ in range(Defines.GRID_NUM)] for _ in range(Defines.GRID_NUM)]
        self.move_history = []
        self.current_player = Defines.BLACK
        self.game_result = None
        self.lock = threading.Lock()  # For thread safety
        
    def make_move(self, move, color):
        """Make a move on the shared board"""
        with self.lock:
            # Validate move first
            if not self.is_valid_move(move):
                raise ValueError(f"Invalid move: {move}")
            
            # Place stones
            pos1 = move.positions[0]
            pos2 = move.positions[1]
            
            if self.board[pos1.x][pos1.y] != Defines.NOSTONE:
                raise ValueError(f"Position ({pos1.x}, {pos1.y}) already occupied")
            if self.board[pos2.x][pos2.y] != Defines.NOSTONE:
                raise ValueError(f"Position ({pos2.x}, {pos2.y}) already occupied")
                
            self.board[pos1.x][pos1.y] = color
            self.board[pos2.x][pos2.y] = color
            
            # Add to history
            self.move_history.append((move, color))
            self.current_player = Defines.BLACK if color == Defines.WHITE else Defines.WHITE
            
            # Check for win
            self.game_result = self.check_win(move, color)
            
    def undo_move(self):
        """Undo the last move"""
        with self.lock:
            if not self.move_history:
                return False
                
            move, color = self.move_history.pop()
            pos1 = move.positions[0]
            pos2 = move.positions[1]
            
            self.board[pos1.x][pos1.y] = Defines.NOSTONE
            self.board[pos2.x][pos2.y] = Defines.NOSTONE
            
            self.current_player = color  # Switch back
            self.game_result = None
            return True
            
    def is_valid_move(self, move):
        """Check if a move is valid"""
        pos1 = move.positions[0]
        pos2 = move.positions[1]
        
        # Check bounds
        if not (0 <= pos1.x < Defines.GRID_NUM and 0 <= pos1.y < Defines.GRID_NUM):
            return False
        if not (0 <= pos2.x < Defines.GRID_NUM and 0 <= pos2.y < Defines.GRID_NUM):
            return False
            
        # Check if positions are empty
        if self.board[pos1.x][pos1.y] != Defines.NOSTONE:
            return False
        if self.board[pos2.x][pos2.y] != Defines.NOSTONE:
            return False
            
        return True
        
    def get_board_copy(self):
        """Get a copy of the current board state"""
        with self.lock:
            return [row[:] for row in self.board]
            
    def get_board_reference(self):
        """Get direct reference to board (use carefully!)"""
        return self.board
        
    def check_win(self, move, color):
        """Check if the last move resulted in a win"""
        # Check both positions of the move
        pos1 = move.positions[0]
        pos2 = move.positions[1]
        
        if self.check_position_win(pos1.x, pos1.y, color) or \
           self.check_position_win(pos2.x, pos2.y, color):
            return color
            
        # Check for draw (board full)
        if self.is_board_full():
            return Defines.DRAW
            
        return None
        
    def check_position_win(self, x, y, color):
        """Check if position (x,y) creates a win for color"""
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        
        for dx, dy in directions:
            count = 1
            # Check positive direction
            nx, ny = x + dx, y + dy
            while (0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and 
                   self.board[nx][ny] == color):
                count += 1
                nx += dx
                ny += dy
                
            # Check negative direction
            nx, ny = x - dx, y - dy
            while (0 <= nx < Defines.GRID_NUM and 0 <= ny < Defines.GRID_NUM and 
                   self.board[nx][ny] == color):
                count += 1
                nx -= dx
                ny -= dy
                
            if count >= 6:
                return True
        return False
        
    def is_board_full(self):
        """Check if board is full"""
        for row in self.board:
            for cell in row:
                if cell == Defines.NOSTONE:
                    return False
        return True
        
    def reset(self):
        """Reset the board"""
        with self.lock:
            self.board = [[Defines.NOSTONE for _ in range(Defines.GRID_NUM)] for _ in range(Defines.GRID_NUM)]
            self.move_history = []
            self.current_player = Defines.BLACK
            self.game_result = None