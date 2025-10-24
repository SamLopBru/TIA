import random
import copy
import numpy as np
from game_engine import GameEngine  # your wrapper that uses SearchEngine

from tools import Defines, check_game_result, StoneMove, StonePosition

def random_weights():
    """Generate a random combination of evaluation weights."""
    w_influence = random.uniform(0.3, 0.7)
    w_pattern = 1.0 - w_influence
    return {"influence": w_influence, "pattern": w_pattern}

def mutate_weights(weights, scale=0.05):
    """Mutate slightly around existing configuration."""
    delta = random.uniform(-scale, scale)
    new_w_infl = weights["influence"] + delta
    new_w_infl = max(0.0, min(1.0, new_w_infl))
    return {"influence": new_w_infl, "pattern": 1.0 - new_w_infl}

def crossover_weights(w1, w2):
    """Average crossover."""
    alpha = random.random()
    infl = alpha * w1["influence"] + (1 - alpha) * w2["influence"]
    return {"influence": infl, "pattern": 1.0 - infl}

def fitness_of_weights(weights, opponent_weights, rounds=3):
    """
    Play a few very short self‑play games between two engines
    and return win ratio as fitness.
    """
    wins = 0
    for _ in range(rounds):
        result = simulate_game(weights, opponent_weights, max_depth=2)
        if result == "A":  # A engine wins
            wins += 1
        elif result == "DRAW":
            wins += 0.5
    return wins / rounds


def simulate_game(weightsA, weightsB, max_depth=2):
    """
    Two lightweight SearchEngines play each other for a few moves.
    """
    from search_engine import SearchEngine
    E1 = SearchEngine(); E1.weights = weightsA
    E2 = SearchEngine(); E2.weights = weightsB

    board = [[0 for _ in range(Defines.GRID_NUM)] for _ in range(Defines.GRID_NUM)]
    color = Defines.BLACK
    last_move = None
    for turn in range(12):  # few turns only, for speed
        engine = E1 if color == Defines.BLACK else E2
        engine.before_search(board, color, max_depth)
        val, move = engine.alpha_beta_pruning(
            board, max_depth, -float("inf"), float("inf"),
            maximizing_player=(color == Defines.BLACK),
            last_move=last_move, max_candidates=15
        )
        if move is None:
            break
        for pos in move.positions:
            board[pos.x][pos.y] = color
        winner = check_game_result(board, move)
        if winner == Defines.BLACK:
            return "A"
        if winner == Defines.WHITE:
            return "B"
        color = Defines.WHITE if color == Defines.BLACK else Defines.BLACK
        last_move = move
    return "DRAW"


def evolve_weights(generations=10, population_size=8):
    population = [random_weights() for _ in range(population_size)]

    for g in range(generations):
        fitness = []
        for w in population:
            opponent = random.choice(population)
            f = fitness_of_weights(w, opponent, rounds=2)
            fitness.append((f, w))

        # Sort by fitness
        fitness.sort(reverse=True, key=lambda t: t[0])
        best = fitness[0][1]
        #print(f"Generation {g}: best weights = {best}")

        # Selection: top 3 survive
        next_gen = [copy.deepcopy(w) for _, w in fitness[:3]]

        # Reproduce until full
        while len(next_gen) < population_size:
            p1, p2 = random.sample(next_gen, 2)
            child = crossover_weights(p1, p2)
            child = mutate_weights(child)
            next_gen.append(child)

        population = next_gen

    print("✅ Final evolved weights:", best)
    return best


if __name__ == "__main__":
    weights_list = []
    for i in range(20):
        random.seed(42 + i)
        np.random.seed(42 + i)
        w = evolve_weights(200, 40)
        weights_list.append(w)
    
    avg_infl = sum(w["influence"] for w in weights_list) / len(weights_list)
    avg_pattern = 1 - avg_infl
    print({"influence": avg_infl, "pattern": avg_pattern})