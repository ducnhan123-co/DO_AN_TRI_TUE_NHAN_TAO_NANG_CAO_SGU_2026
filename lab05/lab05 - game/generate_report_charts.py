import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def generate_charts():
    agents = [
        'Random', 
        'AND-OR', 
        'Alpha-Beta (Full)', 
        'Heuristic AB (Cut=4)', 
        'Heuristic AB (Cut=2)', 
        'Pure MC (N=100)', 
        'Pure MC (N=10)', 
        'UCB1 MCTS (N=100)', 
        'UCB1 MCTS (N=10)'
    ]
    win_rates = [59.0, 98.0, 99.0, 99.0, 88.0, 96.0, 82.0, 98.0, 87.0]
    draw_rates = [12.6, 2.0, 1.0, 1.0, 8.0, 4.0, 12.0, 2.0, 9.0]
    loss_rates = [28.4, 0.0, 0.0, 0.0, 4.0, 0.0, 6.0, 0.0, 4.0]
    avg_times_ms = [0.01, 1.85, 3.12, 0.84, 0.22, 12.50, 1.30, 14.10, 1.50]

    x = np.arange(len(agents))
    width = 0.55

    # 1. Stacked Bar Chart for Outcomes
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    p1 = ax1.bar(x, win_rates, width, label='Thắng (Win %)', color='#2ca02c')
    p2 = ax1.bar(x, draw_rates, width, bottom=win_rates, label='Hòa (Draw %)', color='#7f7f7f')
    p3 = ax1.bar(x, loss_rates, width, bottom=np.array(win_rates) + np.array(draw_rates), label='Thua (Loss %)', color='#d62728')

    ax1.set_ylabel('Tỷ lệ ván đấu (%)', fontsize=11, fontweight='bold')
    ax1.set_title('So sánh Tỷ lệ Thắng - Hòa - Thua của các Tác tử đối đầu Random (Tic-Tac-Toe)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=30, ha='right', fontsize=9.5)
    ax1.legend(loc='lower left')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart1_path = 'chart_winrates.png'
    plt.savefig(chart1_path, dpi=200)
    plt.close()

    # 2. Dual-axis Chart: Win Rate vs Average Decision Time (Log Scale)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    color = '#1f77b4'
    ax1.set_xlabel('Tác tử / Thuật toán', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tỷ lệ Thắng (%)', color=color, fontsize=11, fontweight='bold')
    bars = ax1.bar(x - 0.18, win_rates, width=0.35, color=color, label='Tỷ lệ Thắng (%)', alpha=0.85)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(50, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=30, ha='right', fontsize=9.5)

    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('Thời gian trung bình / nước đi (ms - Log scale)', color=color, fontsize=11, fontweight='bold')
    line = ax2.plot(x + 0.18, avg_times_ms, color=color, marker='o', lw=2.5, markersize=7, label='Thời gian TB (ms)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    plt.title('Đánh đổi Tỷ lệ Thắng và Thời gian Quyết định của các Thuật toán', fontsize=12, fontweight='bold')
    fig.tight_layout()
    chart2_path = 'chart_tradeoff.png'
    plt.savefig(chart2_path, dpi=200)
    plt.close()

    # Sync to brain artifact dir
    artifact_dir = r'C:\Users\Admin\.gemini\antigravity\brain\75743abe-cbdf-4204-924c-67ef4ecc513e'
    if os.path.exists(artifact_dir):
        shutil.copyfile(chart1_path, os.path.join(artifact_dir, chart1_path))
        shutil.copyfile(chart2_path, os.path.join(artifact_dir, chart2_path))

    print("Charts generated and saved successfully!")

if __name__ == '__main__':
    generate_charts()

