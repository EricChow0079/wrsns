# generate_crformer_dataset.py
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
import csv
from scipy.stats import sem, t
from numpy import mean

# 直接导入你原来的类
from Node import Node
from MobileCharger import MobileCharger
from Network import Network


def generate_dataset(num_trajectories=50):
    all_data = []
    df = pd.read_csv("data/thaydoitileguitin.csv")
    index = 0  # 使用第0个场景（你原来的就是这样）

    for traj in range(num_trajectories):
        print(f"\n正在生成第 {traj + 1}/{num_trajectories} 条轨迹...")

        # ==================== 完全复制你Test.py中的建网代码 ====================
        node_pos = list(literal_eval(df.node_pos[index]))
        list_node = []
        for i in range(len(node_pos)):
            location = node_pos[i]
            com_ran = df.commRange[index]
            energy = df.energy[index]
            energy_max = df.energy[index]
            prob = df.freq[index]
            node = Node(location=location, com_ran=com_ran, energy=energy,
                        energy_max=energy_max, id=i, energy_thresh=0.4 * energy, prob=prob)
            list_node.append(node)

        mc = MobileCharger(
            energy=108000,  # 大电池
            capacity=108000,  # 电池容量
            e_move=0.2,  # 移动超省油（原来可能是1~10）
            e_self_charge=0.1,  # 自充效率
            velocity=30  # 30m/s ≈ 108km/h（6倍速！）
        )
        target = [int(item) for item in df.target[index].split(',')]
        net = Network(list_node=list_node, mc=mc, target=target)
        # =====================================================================

        # 关键：不使用任何充电算法，让网络自然崩溃产生级联失效
        net.simulate(optimizer=None)  # 我们的simulate已经改造好

        # ==================== 生成训练样本 ====================
        max_t = min(200, len(net.alive_history))
        T = 10
        future_window = 15

        # 构建每步每个簇的状态历史
        history = []
        for t in range(len(net.alive_history)):
            cluster_stats = {}
            for cid in range(15):
                nodes = [n for n in net.node if getattr(n, 'cluster_id', -1) == cid and n.id != 0]
                if not nodes:
                    continue
                alive = sum(1 for n in nodes if n.is_active)
                energies = [max(n.energy, 0) for n in nodes if n.is_active]  # 防止负值
                loads = [getattr(n, 'load', 0.02) for n in nodes if n.is_active]
                cluster_stats[cid] = {
                    'alive_ratio': alive / len(nodes),
                    'avg_energy': np.mean(energies) if energies else 0,
                    'min_energy': np.min(energies) if energies else 0,
                    'avg_load': np.mean(loads) if loads else 0.02,
                }
            history.append(cluster_stats)

        # 生成样本
        for t in range(T, max_t - future_window,5):
            for cid in range(15):
                if cid not in history[t]:
                    continue

                seq = []
                for past in range(t - T, t):
                    stats = history[past].get(cid,
                                              {'alive_ratio': 0, 'avg_energy': 0, 'min_energy': 0, 'avg_load': 0.02})
                    seq.append([
                        stats['alive_ratio'],
                        stats['avg_energy'] / df.energy[index],  # 归一化
                        stats['min_energy'] / df.energy[index],
                        stats['avg_load']
                    ])

                # 标签：未来15步内该簇死亡≥3个节点 → 高风险
                future_deaths = 0
                for ft in range(t, min(t + future_window, len(net.death_record))):
                    future_deaths += sum(1 for n in net.death_record[ft]
                                         if getattr(n, 'cluster_id', -1) == cid)

                label = 1 if future_deaths >= 3 else 0

                all_data.append({
                    'feature': np.array(seq, dtype=np.float32),  # (10, 4)
                    'label': float(label),
                })

        print(
            f"轨迹 {traj + 1} 完成 | 本轨迹样本数：{len(all_data) - (sum(len(d) for d in all_data[:traj]) if traj > 0 else 0)} | 累计：{len(all_data)}")

    # 保存数据集
    with open('crformer_dataset_500.pkl', 'wb') as f:
        pickle.dump(all_data, f)

    print(f"\n数据集生成完成！共 {len(all_data):,} 条样本")
    print("文件已保存：crformer_dataset_500.pkl")
    print("接下来运行：python train_crformer.py")


if __name__ == '__main__':
    generate_dataset(50)