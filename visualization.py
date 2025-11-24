import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_results(network_history, title="仿真结果"):
    """绘制仿真结果"""
    plt.figure(figsize=(12, 6))

    # 存活节点曲线
    plt.subplot(1, 2, 1)
    plt.plot(network_history['alive_nodes'], 'b-', linewidth=2, label='存活节点')
    plt.xlabel('时间步')
    plt.ylabel('存活节点数')
    plt.title('网络存活情况')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 能量分布
    plt.subplot(1, 2, 2)
    energies = [node.energy for node in network_history['final_nodes'] if node.id != 0]
    plt.hist(energies, bins=20, alpha=0.7, color='green')
    plt.xlabel('节点能量')
    plt.ylabel('节点数量')
    plt.title('最终能量分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison(baseline_history, crformer_history):
    """对比实验结果"""
    plt.figure(figsize=(10, 6))

    plt.plot(baseline_history['alive_nodes'], 'r--', label='无CR-Former', linewidth=2)
    plt.plot(crformer_history['alive_nodes'], 'g-', label='CR-Former', linewidth=2)

    plt.xlabel('时间步')
    plt.ylabel('存活节点数')
    plt.title('CR-Former vs 基线方法')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    plt.show()