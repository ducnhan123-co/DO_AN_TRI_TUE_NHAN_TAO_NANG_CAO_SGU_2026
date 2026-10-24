"""
Script to update n_queens.ipynb with complete solutions for:
- Task 1: Steepest-Ascent Hill Climbing Search
- Task 2: Inefficient Stochastic Hill Climbing
- Task 3: Efficient Stochastic Hill Climbing (First-Choice)
- Task 4: Hill Climbing Search with Random Restarts
- Task 5: Simulated Annealing
"""

import json

def update_notebook():
    with open('n_queens.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Task 1: Cell 13
    nb['cells'][13]['source'] = '''def get_neighbors(board):
    """Generate all n*(n-1) neighbor boards by moving one queen in its column."""
    n = len(board)
    neighbors = []
    for c in range(n):
        curr_row = board[c]
        for r in range(n):
            if r != curr_row:
                nb_board = np.array(board, copy=True)
                nb_board[c] = r
                neighbors.append(nb_board)
    return neighbors

def steepest_ascent_hill_climbing(board):
    """
    Task 1: Steepest-Ascent Hill Climbing Search
    Always selects the neighbor with the lowest conflict count.
    Stops when no neighbor strictly improves the objective.
    """
    curr_board = np.array(board, copy=True)
    curr_f = conflicts(curr_board)
    steps = 0
    history = [curr_f]
    
    while curr_f > 0:
        neighbors = get_neighbors(curr_board)
        best_nb = None
        best_f = curr_f
        for nb_b in neighbors:
            f_nb = conflicts(nb_b)
            if f_nb < best_f:
                best_f = f_nb
                best_nb = nb_b
        if best_f >= curr_f:
            break
        curr_board = best_nb
        curr_f = best_f
        steps += 1
        history.append(curr_f)
        
    return curr_board, curr_f, steps, history

# Demo Task 1
np.random.seed(42)
b4 = random_board(4)
res4, f4, s4, _ = steepest_ascent_hill_climbing(b4)
print(f"4-Queens: start f={conflicts(b4)} -> final f={f4} in {s4} steps: {res4}")

b8 = random_board(8)
res8, f8, s8, _ = steepest_ascent_hill_climbing(b8)
print(f"8-Queens: start f={conflicts(b8)} -> final f={f8} in {s8} steps: {res8}")
'''.splitlines(keepends=True)

    # Task 2: Cell 15
    nb['cells'][15]['source'] = '''def inefficient_stochastic_hill_climbing(board):
    """
    Task 2: Inefficient Stochastic Hill Climbing
    Evaluates all n*(n-1) neighbors, filters improving moves (uphill/strictly lower conflicts),
    and randomly selects one improving move.
    """
    curr_board = np.array(board, copy=True)
    curr_f = conflicts(curr_board)
    steps = 0
    history = [curr_f]
    
    while curr_f > 0:
        neighbors = get_neighbors(curr_board)
        improving_moves = []
        for nb_b in neighbors:
            f_nb = conflicts(nb_b)
            if f_nb < curr_f:
                improving_moves.append((nb_b, f_nb))
        if not improving_moves:
            break
        idx = np.random.randint(0, len(improving_moves))
        curr_board, curr_f = improving_moves[idx]
        steps += 1
        history.append(curr_f)
        
    return curr_board, curr_f, steps, history

# Demo Task 2
np.random.seed(42)
b4 = random_board(4)
res4, f4, s4, _ = inefficient_stochastic_hill_climbing(b4)
print(f"4-Queens: start f={conflicts(b4)} -> final f={f4} in {s4} steps: {res4}")

b8 = random_board(8)
res8, f8, s8, _ = inefficient_stochastic_hill_climbing(b8)
print(f"8-Queens: start f={conflicts(b8)} -> final f={f8} in {s8} steps: {res8}")
'''.splitlines(keepends=True)

    # Task 3: Cell 18
    nb['cells'][18]['source'] = '''def get_random_neighbor(board):
    """Generate a single random neighbor in O(1)."""
    n = len(board)
    c = np.random.randint(0, n)
    curr_row = board[c]
    new_row = np.random.randint(0, n - 1)
    if new_row >= curr_row:
        new_row += 1
    nb_board = np.array(board, copy=True)
    nb_board[c] = new_row
    return nb_board

def efficient_stochastic_hill_climbing(board, max_fails=None):
    """
    Task 3: Efficient Stochastic Hill Climbing (First-Choice)
    Generates a single random neighbor at a time, immediately accepts if strictly better.
    Stopping criterion: stops if not able to improve during the last x tries.
    Here x = max(50, 3 * n * (n - 1)), derived from the Coupon Collector's Problem.
    """
    n = len(board)
    if max_fails is None:
        max_fails = max(50, 3 * n * (n - 1))
        
    curr_board = np.array(board, copy=True)
    curr_f = conflicts(curr_board)
    steps = 0
    fail_count = 0
    history = [curr_f]
    max_total_trials = 10000
    
    while curr_f > 0 and fail_count < max_fails and steps < max_total_trials:
        nb_b = get_random_neighbor(curr_board)
        f_nb = conflicts(nb_b)
        if f_nb < curr_f:
            curr_board = nb_b
            curr_f = f_nb
            steps += 1
            fail_count = 0
            history.append(curr_f)
        else:
            fail_count += 1
            
    return curr_board, curr_f, steps, history

# Demo Task 3
np.random.seed(42)
b4 = random_board(4)
res4, f4, s4, _ = efficient_stochastic_hill_climbing(b4)
print(f"4-Queens: start f={conflicts(b4)} -> final f={f4} in {s4} steps: {res4}")

b8 = random_board(8)
res8, f8, s8, _ = efficient_stochastic_hill_climbing(b8)
print(f"8-Queens: start f={conflicts(b8)} -> final f={f8} in {s8} steps: {res8}")
'''.splitlines(keepends=True)

    # Task 4: Cell 20
    nb['cells'][20]['source'] = '''import time

def random_restart_hill_climbing(algorithm_func, n, max_restarts=100, **kwargs):
    """
    Task 4: Hill Climbing Search with Random Restarts
    Restarts the algorithm up to 100 times with a random board.
    Stops early as soon as an optimal solution (conflicts == 0) is found.
    Returns the best board, best f, restarts used, and total steps.
    """
    start_time = time.time()
    best_board = None
    best_f = float('inf')
    total_steps = 0
    restarts_used = 0
    
    for r in range(1, max_restarts + 1):
        restarts_used = r
        init_board = random_board(n)
        final_board, final_f, steps, _ = algorithm_func(init_board, **kwargs)
        total_steps += steps
        
        if final_f < best_f:
            best_f = final_f
            best_board = final_board
            
        if best_f == 0:
            break
            
    elapsed = time.time() - start_time
    return {
        'best_board': best_board,
        'best_f': best_f,
        'restarts_used': restarts_used,
        'total_steps': total_steps,
        'success': (best_f == 0),
        'runtime': elapsed
    }

# Run Random Restarts (max 100) for all three Hill Climbing variants on n=4 and n=8
np.random.seed(42)
print("=" * 75)
print("TASK 4: HILL CLIMBING WITH RANDOM RESTARTS (MAX 100 RESTARTS)")
print("=" * 75)

algorithms = [
    ("Steepest-Ascent HC", steepest_ascent_hill_climbing, {}),
    ("Inefficient Stochastic HC", inefficient_stochastic_hill_climbing, {}),
    ("Efficient Stochastic HC", efficient_stochastic_hill_climbing, {})
]

for n in [4, 8]:
    print(f"\\n--- Board Size n = {n} ---")
    for name, func, kwargs in algorithms:
        successes = 0
        restarts_list = []
        runtimes = []
        for _ in range(20):
            res = random_restart_hill_climbing(func, n=n, max_restarts=100, **kwargs)
            if res['success']:
                successes += 1
            restarts_list.append(res['restarts_used'])
            runtimes.append(res['runtime'])
        
        rate = (successes / 20) * 100
        avg_r = np.mean(restarts_list)
        avg_t = np.mean(runtimes) * 1000
        print(f"{name:<28}: Success Rate = {rate:5.1f}%, Avg Restarts = {avg_r:4.1f}, Avg Time = {avg_t:6.2f} ms")
'''.splitlines(keepends=True)

    # Task 5: Cell 22
    nb['cells'][22]['source'] = '''import math

def simulated_annealing(n, initial_temp=15.0, alpha=0.992, max_iter=3000, min_temp=1e-4, initial_board=None):
    """
    Task 5: Simulated Annealing for the n-Queens Problem.
    Acceptance probability: Delta = f(new) - f(curr)
    - If Delta < 0: Accept (strictly better).
    - If Delta == 0: Accept (sideways move, p = exp(0) = 1).
    - If Delta > 0: Accept with probability p = exp(-Delta / T).
    Cooling schedule: T_{k+1} = T_k * alpha (Geometric / Exponential schedule).
    """
    start_time = time.time()
    if initial_board is None:
        curr_board = random_board(n)
    else:
        curr_board = np.array(initial_board, copy=True)
        
    curr_f = conflicts(curr_board)
    best_board = np.array(curr_board, copy=True)
    best_f = curr_f
    T = float(initial_temp)
    
    conflict_history = [curr_f]
    best_f_history = [best_f]
    temp_history = [T]
    accepted_worse = 0
    it = 0
    
    while it < max_iter and T > min_temp and best_f > 0:
        it += 1
        nb_b = get_random_neighbor(curr_board)
        f_nb = conflicts(nb_b)
        delta = f_nb - curr_f
        
        accept = False
        if delta <= 0:
            accept = True
        else:
            p = math.exp(-delta / T)
            if np.random.rand() < p:
                accept = True
                accepted_worse += 1
                
        if accept:
            curr_board = nb_b
            curr_f = f_nb
            if curr_f < best_f:
                best_f = curr_f
                best_board = np.array(curr_board, copy=True)
                
        conflict_history.append(curr_f)
        best_f_history.append(best_f)
        temp_history.append(T)
        T *= alpha
        
    elapsed = time.time() - start_time
    return {
        'best_board': best_board,
        'best_f': best_f,
        'iterations': it,
        'success': (best_f == 0),
        'conflict_history': conflict_history,
        'best_f_history': best_f_history,
        'temp_history': temp_history,
        'accepted_worse': accepted_worse,
        'runtime': elapsed
    }

# 1. Visualization of Search Process (Line chart of conflicts & temperature over iterations)
np.random.seed(42)
res_sa = simulated_annealing(8, initial_temp=20.0, alpha=0.995, max_iter=3000)
print(f"Simulated Annealing on 8-Queens: Success={res_sa['success']}, Final Conflicts={res_sa['best_f']} in {res_sa['iterations']} iterations.")
print(f"Optimal Board: {res_sa['best_board']}")

fig, ax1 = plt.subplots(figsize=(10, 5))
color1 = '#1f77b4'
color2 = '#2ca02c'
color3 = '#d62728'

ax1.set_xlabel('Iterations', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Conflicts f(board)', color=color1, fontsize=11, fontweight='bold')
l1 = ax1.plot(res_sa['conflict_history'], color=color1, alpha=0.5, label='Current Conflicts')
l2 = ax1.plot(res_sa['best_f_history'], color=color2, linewidth=2, label='Best Conflicts So Far')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel('Temperature T', color=color3, fontsize=11, fontweight='bold')
l3 = ax2.plot(res_sa['temp_history'], color=color3, linestyle='--', label='Temperature T')
ax2.tick_params(axis='y', labelcolor=color3)

lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')
plt.title(f"Simulated Annealing Search Process (8-Queens) - Result: f={res_sa['best_f']}", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. Experiment with different annealing schedules (alpha)
print("\\n" + "=" * 65)
print("EXPERIMENT WITH ANNEALING SCHEDULES (8-Queens, 30 runs each)")
print("=" * 65)
alphas = [0.90, 0.95, 0.98, 0.99, 0.995]
for a in alphas:
    succ = sum(1 for _ in range(30) if simulated_annealing(8, initial_temp=15.0, alpha=a, max_iter=2500)['success'])
    print(f"Cooling Rate alpha = {a:.3f}: Success Rate = {(succ/30)*100:5.1f}%")
'''.splitlines(keepends=True)

    with open('n_queens.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Updated n_queens.ipynb successfully!")

if __name__ == '__main__':
    update_notebook()
