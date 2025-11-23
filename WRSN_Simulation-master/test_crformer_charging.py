# test_crformer_charging.py  ——  见证屠杀的时刻！
import torch
from my_crformer import CRFormer  # 或者把模型类直接写这里
from Network import Network

# ... 其他导入同之前

model = CRFormer()
model.load_state_dict(torch.load('crformer_750_best.pth'))
model.eval()
print("CR-Former 加载成功！准备屠杀级联失效！")


def crformer_priority(net):
    """用CR-Former计算每个簇的级联风险分数"""
    with torch.no_grad():
        risks = []
        for cid in range(15):
            nodes = [n for n in net.node if getattr(n, 'cluster_id', -1) == cid and n.id != 0]
            if not nodes:
                risks.append(0.0)
                continue

            # 构造最近10步的特征序列
            seq = []
            current_t = len(net.alive_history) - 1
            for past in range(current_t - 9, current_t + 1):
                if past < 0:
                    seq.append([1.0, 1.0, 1.0, 0.02])
                    continue
                alive = sum(1 for n in nodes if n.is_active)
                energies = [max(n.energy, 0) for n in nodes if n.is_active]
                loads = [getattr(n, 'load', 0.02) for n in nodes if n.is_active]
                seq.append([
                    alive / len(nodes),
                    (np.mean(energies) / 100 if energies else 1.0),
                    (np.min(energies) / 100 if energies else 1.0),
                    np.mean(loads) if loads else 0.02
                ])

            feature = torch.FloatTensor([seq])  # (1,10,4)
            risk = model(feature).item()
            risks.append(risk)

    return risks  # 15个簇的风险分数，越大越危险


# 运行对比实验
print("不加CR-Former（原始崩溃）".center(50, "="))
net1 = Network(...)  # 你原来的建网代码
net1.simulate(optimizer=None)

print("加入CR-Former智能充电".center(50, "="))


class CRFormerOptimizer:
    def __init__(self, model):
        self.model = model
        self.cooldown = {}  # 节点冷却字典

    def select_target(self, net):
        # 每步减冷却
        for nid in list(self.cooldown.keys()):
            self.cooldown[nid] -= 1
            if self.cooldown[nid] <= 0:
                del self.cooldown[nid]

        # 1. 获取风险
        risks = get_risk(net)  # 你之前写好的函数
        if max(risks) < 0.6:
            return None

        best_cid = np.argmax(risks)
        # 2. 选候选节点（同簇 + 存活 + 未冷却）
        candidates = [n for n in net.node
                      if getattr(n, 'cluster_id', -1) == best_cid
                      and n.is_active and n.id != 0
                      and n.id not in self.cooldown]
        if not candidates:
            return None

        # 3. 选能量最低的
        target = min(candidates, key=lambda n: n.energy)
        old_energy = target.energy

        # 4. 发出救援指令！
        print(f"第{net.time_step if hasattr(net, 'time_step') else '?'}步 | "
              f"CR-Former指挥：去救簇{best_cid}的节点{target.id}！"
              f"风险{np.max(risks):.3f} 能量{old_energy:.1f}J")

        # 5. 加入冷却（防止一直救同一个）
        self.cooldown[target.id] = 8

        return target


opt = CRFormerOptimizer()
net2 = Network(...)  # 重新建网
net2.simulate(optimizer=opt)