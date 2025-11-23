import csv

from scipy.spatial import distance

import Parameter as para
from Network_Method import uniform_com_func, to_string, count_package_function


class Network:
    def __init__(self, list_node=None, mc=None, target=None):
        self.node = list_node
        self.set_neighbor()
        self.set_level()
        self.mc = mc
        self.target = target

    def set_neighbor(self):
        for node in self.node:
            for other in self.node:
                if other.id != node.id and distance.euclidean(node.location, other.location) <= node.com_ran:
                    node.neighbor.append(other.id)

    def set_level(self):
        queue = []
        for node in self.node:
            if distance.euclidean(node.location, para.base) < node.com_ran:
                node.level = 1
                queue.append(node.id)
        while queue:
            for neighbor_id in self.node[queue[0]].neighbor:
                if not self.node[neighbor_id].level:
                    self.node[neighbor_id].level = self.node[queue[0]].level + 1
                    queue.append(neighbor_id)
            queue.pop(0)

    def communicate(self, func=uniform_com_func):
        return func(self)

    def run_per_second(self, t, optimizer):
        state = self.communicate()
        request_id = []
        for index, node in enumerate(self.node):
            if node.energy < node.energy_thresh:
                node.request(mc=self.mc, t=t)
                request_id.append(index)
            else:
                node.is_request = False
        if request_id:
            for index, node in enumerate(self.node):
                if index not in request_id and (t - node.check_point[-1]["time"]) > 50:
                    node.set_check_point(t)
        if optimizer:
            self.mc.run(network=self, time_stem=t, net=self, optimizer=optimizer)
        return state

    def simulate_lifetime(self, optimizer, file_name="log/energy_log.csv"):
        energy_log = open(file_name, "w")
        writer = csv.DictWriter(energy_log, fieldnames=["time", "mc energy", "min energy"])
        writer.writeheader()
        t = 0
        while self.node[self.find_min_node()].energy >= 0:
            t = t + 1
            if (t-1) % 100 == 0:
                print(t, self.mc.current, self.node[self.find_min_node()].energy)
            state = self.run_per_second(t, optimizer)
            if not (t - 1) % 50:
                writer.writerow(
                    {"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})
        print(t, self.mc.current, self.node[self.find_min_node()].energy)
        writer.writerow({"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})
        energy_log.close()
        return t

    def simulate_max_time(self, optimizer, max_time=10000, file_name="log/information_log.csv"):
        information_log = open(file_name, "w")
        writer = csv.DictWriter(information_log, fieldnames=["time", "nb dead", "nb package"])
        writer.writeheader()
        nb_dead = 0
        nb_package = len(self.target)
        t = 0
        while t <= max_time:
            t += 1
            if (t-1)%100 == 0:
                print(t, self.mc.current, self.node[self.find_min_node()].energy)
            state = self.run_per_second(t, optimizer)
            current_dead = self.count_dead_node()
            current_package = self.count_package()
            if current_dead != nb_dead or current_package != nb_package:
                nb_dead = current_dead
                nb_package = current_package
                writer.writerow({"time": t, "nb dead": nb_dead, "nb package": nb_package})
        print(t, self.mc.current, self.node[self.find_min_node()].energy)
        information_log.close()
        return t

    def simulate(self, optimizer, file_name=None):
        import warnings
        warnings.filterwarnings("ignore")
        import os
        os.environ['OMP_NUM_THREADS'] = '2'
        import numpy as np
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 分簇代码保持不变...
        # ...（你原来的分簇代码）

        time_step = 0
        alive_history = []
        cascade_events = 0
        self.death_record = []
        self.alive_history = alive_history

        while time_step < 10000:
            time_step += 1
            dead_this_step = []
            self.death_record.append([])

            # 1. 所有节点耗能
            for node in self.node:
                if node.id == 0 or not node.is_active: continue
                consume = node.load * 20
                node.energy -= consume
                node.used_energy += consume
                if hasattr(optimizer, 'consume_energy'):
                    node.energy -= optimizer.consume_energy(node)

            # 2. CR-Former提前干预！（唯一充电点！）
            if optimizer is not None and hasattr(optimizer, 'select_target'):
                rescue_node = optimizer.select_target(self)
                if rescue_node is not None:
                    old_energy = rescue_node.energy
                    rescue_node.energy = rescue_node.energy_max
                    print(
                        f"第{time_step}步 | CR-Former神级提前救援 → 节点{rescue_node.id} {old_energy:.1f}→满电！阻止级联！")

            # 3. 死亡判断 + 级联失效
            for node in self.node:
                if node.id == 0 or not node.is_active: continue
                if node.energy <= node.energy_thresh:
                    node.is_active = False
                    dead_this_step.append(node)
                    self.death_record[-1].append(node)

            for dead_node in dead_this_step:
                alive_mates = [n for n in self.node
                               if n.cluster_id == dead_node.cluster_id and n.is_active and n.id != 0]
                if alive_mates:
                    extra_load = dead_node.load * 1.5
                    for mate in alive_mates:
                        mate.load += extra_load / len(alive_mates)
                    cascade_events += 1
                    print(f"级联失效触发！时间{time_step} | 节点{dead_node.id}死亡 | 簇{dead_node.cluster_id}")

            # 删除你原来的第二个充电块！！！

            alive_count = sum(1 for n in self.node if n.is_active and n.id != 0)
            alive_history.append(alive_count)

            if alive_count < len(self.node) * 0.1:
                print(f"网络崩溃！存活节点仅剩 {alive_count}，提前终止")
                break

        # 出图代码保持不变...
        plt.figure(figsize=(12, 7))
        plt.plot(alive_history, color='#d62728', linewidth=3)
        plt.title('WRSN 密集非均衡网络中的级联失效现象\n（CR-Former干预后）', fontsize=16)
        plt.xlabel('时间步')
        plt.ylabel('存活节点数')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("cascade_failure_final.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"仿真结束 | 总步数 {time_step} | 最终存活 {alive_count} | 级联事件 {cascade_events} 次")
        return time_step

    def print_net(self, func=to_string):
        func(self)

    def find_min_node(self):
        min_energy = 10 ** 10
        min_id = -1
        for node in self.node:
            if node.energy < min_energy:
                min_energy = node.energy
                min_id = node.id
        return min_id

    def count_dead_node(self):
        count = 0
        for node in self.node:
            if node.energy < 0:
                count += 1
        return count

    def count_package(self, count_func=count_package_function):
        count = count_func(self)
        return count
