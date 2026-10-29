import os
from connect4 import Connect4
from heuristic import HeuristicAgent
from minimax import MinimaxAgent
from ensemble import EnsembleAgent
from stable_baselines3 import PPO
from gui import Connect4GUI

WIN_REWARD = 10
BLOCK_REWARD = 3
DRAW_REWARD = 1
CONTINUE_REWARD = 0
ILLEGAL_MOVE_PENALTY = -10


def play(opponent=None, player=None):
    """
    Khởi chạy trò chơi Connect 4:
    - Nếu player is None: Mở giao diện đồ họa Pygame để Người chơi đấu với AI (opponent).
    - Nếu có player (Agent vs Agent): Cho 2 AI đấu mô phỏng trên console / stdout.
    """
    if opponent is None:
        opponent = HeuristicAgent(c=0)

    # Người chơi (Human vs AI qua giao diện Pygame)
    if player is None:
        gui = Connect4GUI(opponent=opponent, player=None)
        gui.run()
    else:
        # AI vs AI (Console mode)
        env = Connect4()
        env.set_opp_agent(opponent, 0)
        env.render(mode="stdout")
        symb1, symb2 = env.reset()[1]["symbols"]
        done = False
        while not done:
            env.render(mode="stdout")
            action = player.predict(env.grid)[0]
            _, reward, done, _, _ = env.step(action)
        env.render(mode="stdout")
        print("Player's symbol:", symb1, "\nAI's symbol:", symb2)
        if reward == WIN_REWARD:
            print("Player wins!")
        elif reward == DRAW_REWARD:
            print("Draw!")
        else:
            print("AI wins!")
        env.close()


def load_best_agent():
    """Hàm tiện ích nạp EnsembleAgent gồm 15 PPO agent đã huấn luyện."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(base_dir, "saved_agents")

    print("Đang nạp các mô hình PPO vào EnsembleAgent, vui lòng đợi giây lát...")
    players = [
        PPO.load(os.path.join(agents_dir, f"agent{i}"))
        for i in range(15)
        if os.path.exists(os.path.join(agents_dir, f"agent{i}.zip"))
    ]

    if len(players) > 0:
        print(f"-> Đã nạp thành công {len(players)} agent PPO vào EnsembleAgent!")
        return EnsembleAgent(players)
    else:
        print("-> Không tìm thấy agent trong saved_agents/, sử dụng HeuristicAgent dự phòng.")
        return HeuristicAgent(c=0)


if __name__ == "__main__":
    # Nạp AI mạnh nhất (Ensemble PPO)
    ai_opponent = load_best_agent()

    # Có thể chọn các đối thủ khác như:
    # ai_opponent = HeuristicAgent(c=0)
    # ai_opponent = PPO.load("./saved_agents/agent15")
    # ai_opponent = MinimaxAgent(agent_symb=2, opp_symb=1)

    # Bắt đầu chơi
    play(opponent=ai_opponent)
