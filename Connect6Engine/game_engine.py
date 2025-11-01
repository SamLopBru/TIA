from defines import *
from tools import init_board, make_move, check_game_result, msg2move, move2msg, print_board, log_to_file
import sys
from search_engine import SearchEngine
import time

import cProfile
import pstats

class GameEngine:
    def __init__(self, name=Defines.ENGINE_NAME, ):
        if name and len(name) > 0:
            if len(name) < Defines.MSG_LENGTH:
                self.m_engine_name = name
            else:
                print(f"Too long Engine Name: {name}, should be less than: {Defines.MSG_LENGTH}")
        
        self.m_alphabeta_depth = 4
        self.m_search_engine = SearchEngine()
        self.m_best_move = StoneMove()
        self.initialize = False
        self.metrics = None
            
        self.m_board = [[0]*Defines.GRID_NUM for i in range(Defines.GRID_NUM)]
        self.init_game()

    def init_game(self):
        init_board(self.m_board)

    def on_help(self):
        print(
            f"On help for GameEngine {self.m_engine_name}\n"
            " name        - print the name of the Game Engine.\n"
            " print       - print the board.\n"
            " exit/quit   - quit the game.\n"
            " black XXXX  - place the black stone on the position XXXX on the board.\n"
            " white XXXX  - place the white stone on the position XXXX on the board, X is from A to S.\n"
            " next        - the engine will search the move for the next step.\n"
            " move XXXX   - tell the engine that the opponent made the move XXXX,\n"
            "              and the engine will search the move for the next step.\n"
            " new black   - start a new game and set the engine to black player.\n"
            " new white   - start a new game and set it to white.\n"
            " depth d     - set the alpha beta search depth, default is 6.\n"
            " vcf         - set vcf search.\n"
            " unvcf       - set none vcf search.\n"
            " help        - print this help.\n")

    def run(self):
        msg = ""
        self.on_help()
        while True:
            msg = input().strip()
            log_to_file(msg)
            if msg == "name":
                print(f"name {self.m_engine_name}")
            elif msg == "exit" or msg == "quit":
                break
            elif msg == "print":
                print_board(self.m_board, self.m_best_move)
            elif msg == "vcf":
                self.m_vcf = True
            elif msg == "unvcf":
                self.m_vcf = False
            elif msg.startswith("black"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.BLACK)
                self.m_chess_type = Defines.BLACK
            elif msg.startswith("white"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.WHITE)
                self.m_chess_type = Defines.WHITE
            elif msg == "next":
                self.m_chess_type = self.m_chess_type ^ 3
                if self.search_a_move(self.m_chess_type, self.m_best_move):
                    make_move(self.m_board, self.m_best_move, self.m_chess_type)
                    msg = f"move {move2msg(self.m_best_move)}"
                    print(msg)
                    flush_output()
            elif msg.startswith("new"):
                self.init_game()

                if msg[4:] == "black":
                    self.m_best_move = msg2move("JJ")
                    self.initialize = True
                    make_move(self.m_board, self.m_best_move, Defines.BLACK)
                    self.m_chess_type = Defines.BLACK
                    msg = "move JJ"
                    print(msg)
                    flush_output()
                else:
                    self.m_chess_type = Defines.WHITE

            elif msg.startswith("move"):
                self.m_best_move = msg2move(msg[5:])
                make_move(self.m_board, self.m_best_move, self.m_chess_type ^ 3)

                # official win/draw check after opponent move
                result = check_game_result(self.m_board, self.m_best_move)
                if result == Defines.BLACK:
                    print("Black wins!")
                    continue
                elif result == Defines.WHITE:
                    print("White wins!")
                    continue
                elif result == Defines.DRAW:
                    print("Game ends in a draw!")
                    continue

                # continue AI response if still playing
                if self.search_a_move(self.m_chess_type, self.m_best_move):
                    msg = f"move {move2msg(self.m_best_move)}"
                    make_move(self.m_board, self.m_best_move, self.m_chess_type)

                    # check result also after AI move
                    result = check_game_result(self.m_board, self.m_best_move)
                    if result == Defines.BLACK:
                        print("Black wins!")
                    elif result == Defines.WHITE:
                        print("White wins!")
                    elif result == Defines.DRAW:
                        print("Game ends in a draw!")

                    print(msg)
                    flush_output()

            elif msg.startswith("depth"):
                d = int(msg[6:])
                if 0 < d < 10:
                    self.m_alphabeta_depth = d
                print(f"Set the search depth to {self.m_alphabeta_depth}.\n")

            elif msg == "help":
                self.on_help()
        return 0


    def search_a_move(self, ourColor, bestMove):
        """
        Uses alpha-beta pruning to find the best move for the current color.
        """
        depth = self.m_alphabeta_depth
        maximizing = (ourColor == Defines.BLACK)
        
        # Handle first move for BLACK
        if not self.initialize and ourColor == Defines.BLACK:
            center = Defines.GRID_NUM // 2
            bestMove.positions[0].x = center
            bestMove.positions[0].y = center
            bestMove.positions[1].x = center
            bestMove.positions[1].y = center
            bestMove.color = ourColor
            self.initialize = True
            return True

        # Run alpha-beta search
        start_time = time.perf_counter()
        
        score, move = self.m_search_engine.alpha_beta_pruning(  # ✅ Only unpack 2 values
            self.m_board,
            depth,
            -float('inf'),
            float('inf'),
            maximizing,
            bestMove,
            max_candidates=40,
            is_root=True
        )

        self.metrics = self.m_search_engine.metrics
        
        # ✅ Access metrics directly from the engine
        decision_time = time.perf_counter() - start_time
        self.m_search_engine.metrics['decision_time'] = decision_time
        
        self.print_search_metrics(self.m_search_engine.metrics, decision_time)

        if move:
            pos1 = move.positions[0]
            pos2 = move.positions[1]

            bestMove.positions[0].x = pos1.x
            bestMove.positions[0].y = pos1.y
            bestMove.positions[1].x = pos2.x
            bestMove.positions[1].y = pos2.y
            bestMove.color = ourColor

            print(f"Alpha-Beta Score = {score}, BestMove = ({pos1.x},{pos1.y}) & ({pos2.x},{pos2.y})")
            return True
        else:
            print("No valid move found.")
            return False
    
    @staticmethod
    def print_search_metrics(metrics, decision_time):
        """Print metrics after a search completes"""
        print(f"\n{'='*50}")
        print(f"SEARCH METRICS")
        print(f"{'='*50}")
        print(f"Decision time:        {decision_time:.3f}s")
        print(f"Nodes expanded:       {metrics['nodes_expanded']:,}")
        print(f"Nodes pruned:         {metrics['nodes_pruned']:,}")
        print(f"Transposition hits:   {metrics['transposition_hits']:,}")
        print(f"Quiescence calls:     {metrics['quiescence_calls']:,}")
        print(f"Max depth reached:    {metrics['max_depth_reached']}")
        
        # Calculate pruning efficiency
        total_nodes = metrics['nodes_expanded'] + metrics['nodes_pruned']
        if total_nodes > 0:
            prune_rate = (metrics['nodes_pruned'] / total_nodes) * 100
            print(f"Pruning efficiency:   {prune_rate:.1f}%")
        
        # Nodes per second
        if decision_time > 0:
            nps = metrics['nodes_expanded'] / decision_time
            print(f"Nodes per second:     {nps:,.0f}")
    

    def run_profiling(self):
        engine = SearchEngine()
        # Setup dummy board and parameters as needed:
        board = self.m_board
        depth = 5    # Example depth

        profiler = cProfile.Profile()
        profiler.enable()
        engine.alpha_beta_pruning(board, depth, -float('inf'), float('inf'), True, None, is_root=True)
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(30)  # Show top 30 time-consuming functions


def flush_output():
    sys.stdout.flush()

# Create an instance of GameEngine and run the game
if __name__ == "__main__":
    game_engine = GameEngine()
    game_engine.run()
