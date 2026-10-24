# Connect4 Reinforcement Learning Project

## Objective
Develop a Reinforcement Learning (RL) agent capable of effectively playing Connect 4 by:
1. Implementing and testing various RL algorithms.
2. Comparing the performance of these approaches.

## About Connect 4
Connect 4 is a two-player, turn-based strategy game. The objective is to be the first to connect four of your pieces vertically, horizontally, or diagonally.

### Game Rules:
- **Board Dimensions**: 6 rows × 7 columns.
- **Players**: Each player uses distinct tokens (e.g., red and yellow).
- **Turns**: Players take turns dropping tokens into columns.
- **Winning**: Form a line of four tokens.
- **Draw**: If the board is full and no one connects four.

## State Representation
- **Game Board**: Represented as a 6 × 7 grid with 42 blocks.
  - `0`: Empty block.
  - `1`: Filled by Player 1.
  - `2`: Filled by Player 2.
- **Actions**:
  - **Action Space**: 7 discrete actions (columns 0-6).
  - Invalid actions (e.g., selecting a full column) incur penalties.

## Reward Shaping
- **Winning Move**: +10
- **Blocking Opponent**: +3
- **Illegal Move**: -50
- **Draw**: 0
- **Ongoing Game**: 0

## Environment Overview
The Connect 4 environment is implemented as a custom Gym environment. Key features include:
- **Action Space**: Seven discrete actions representing columns.
- **Observation Space**: A 6 × 7 grid indicating the board state.
- **Pygame Rendering**: The game board is visually rendered using Pygame.
- **Random Opponent**: By default, the opponent is random, but it can be configured with heuristic or any other trained agents.
- **Reward System**: Rewards and penalties guide the agent's learning process.

## Heuristic Agent
The Heuristic Agent employs domain-specific knowledge to make decisions. Key features include:
- **Winning Move Priority**: Prioritizes actions leading to an immediate win.
- **Blocking Opponent**: Detects and blocks the opponent's winning moves.
- **Central Column Preference**: Encourages moves near the center column for strategic positioning.
- **Upper Confidence Bound (UCB)**: Balances exploration and exploitation during gameplay.
- **Scoring Mechanism**: Evaluates potential moves based on heuristics such as chain formation and board evaluation.

## Minimax Agent
The Minimax Agent uses a game-tree search algorithm with the following features:
- **Recursive Search**: Evaluates potential moves up to a configurable depth.
- **Alpha-Beta Pruning**: Optimizes the search by eliminating branches that do not influence the final decision.
- **Winning and Blocking Detection**: Identifies and prioritizes winning or blocking moves.
- **Terminal State Evaluation**: Determines outcomes such as wins, losses, or draws at leaf nodes.
- **Strategic Play**: Calculates optimal moves for both the agent and opponent to maximize the agent's chances of winning.



<div align="center">
    <img src="assets/play_against_heurestic_agent.gif" alt="Connect 4 Gameplay" width="600">
    <p><em>Figure: Gameplay against Heurestic Agent</em></p>
</div>


## Using Play, Test, and Train Functions
### Play
The `play` function allows you to interact with the Connect 4 environment. You can either play as a human against an agent or have two agents compete.
- **Parameters**:
  - `opponent`: The agent you want to play against.
  - `player` (optional): An agent to play as Player 1 (default is human input).
- **How to Run**:
  - Human vs. Agent: Run `play(agent)`.
  - Agent vs. Agent: Provide both `player` and `opponent` agents.

### Test
The `test` function evaluates an agent's performance against an opponent over multiple episodes.
- **Parameters**:
  - `player`: The agent to be tested.
  - `opponent` (optional): The agent acting as the opponent (default is random).
  - `mode`: Rendering mode (e.g., "human" or "stdout").
- **How to Run**:
  - Use `test(player, opponent)` to compute win rates.
- **Example**:
  - Evaluate an ensemble agent against random opponents:
    ```python
    players = [PPO.load(f"./saved_agents/agent{i}") for i in range(15)]
    ensemble_player = EnsembleAgent(players)
    NUM_GAMES = 10000
    wins = 0
    for _ in range(NUM_GAMES):
        wins += test(ensemble_player)
    winrate = wins / NUM_GAMES
    print("Win rate:", winrate)
    ```

### Train
The `train` function is used to train an RL agent on the Connect 4 environment.
- **Parameters**:
  - `agent_algo`: The RL algorithm to use (e.g., PPO, DQN, A2C).
  - `total_timesteps`: Total timesteps for training.
  - `opp_path` (optional): Path to a pre-trained opponent agent.
  - `opp_epsilon`: Exploration factor for the opponent.
  - `tb_dir`: Directory for TensorBoard logs.
  - `tb_log_name`: Name for TensorBoard logs.
- **How to Run**:
  - Train a PPO agent:
    ```python
    agent = train(agent_algo="PPO", total_timesteps=100_000, tb_dir="./logs", tb_log_name="PPO_agent")
    ```
  - Train an agent with a pre-trained opponent:
    ```python
    agent = train(agent_algo="PPO", total_timesteps=100_000, opp_path="./saved_agents/agent0", opp_algo="PPO", tb_dir="./logs", tb_log_name="PPO_vs_agent0")
    ```


## Libraries Used
- **[Gymnasium](https://gymnasium.farama.org/)**: For creating the Connect 4 environment.
- **[Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/)**: For RL algorithms such as PPO, DQN, and A2C.
- **[Pygame](https://www.pygame.org/)**: For UI development and gameplay visualization.
- **[tqdm](https://tqdm.github.io/)**: For progress bar visualization during training and evaluation.

## How to Run
1. Install dependencies: `pip install gymnasium stable-baselines3 pygame`.
2. Clone the repository and navigate to the project directory.
3. Run training scripts for specific algorithms.
4. Visualize results using TensorBoard.
