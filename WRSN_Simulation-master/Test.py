from numpy import mean
from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger
from Q__Learning import Q_learning
from Inma import Inma
import csv
from scipy.stats import sem, t






df = pd.read_csv("data/thaydoitileguitin.csv")
for index in range(1):
    chooser_alpha = open("log/q_learning_confident3.csv", "w")
    result = csv.DictWriter(chooser_alpha, fieldnames=["nb run", "lifetime"])
    result.writeheader()
    life_time = []
    for nb_run in range(10):
        random.seed(nb_run)

        node_pos = list(literal_eval(df.node_pos[index]))
        list_node = []
        for i in range(len(node_pos)):
            location = node_pos[i]
            com_ran = df.commRange[index]
            energy = df.energy[index]
            energy_max = df.energy[index]
            prob = df.freq[index]
            node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                        energy_thresh=0.4 * energy, prob=prob)
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
        print(len(net.node), len(net.target), max(net.target))
        q_learning = Q_learning(network=net)
        # inma = Inma()
        file_name = "log/q_learning_" + str(index) + ".csv"
        temp = net.simulate(optimizer=q_learning, file_name=file_name)
        life_time.append(temp)
        result.writerow({"nb run": nb_run, "lifetime": temp})

    confidence = 0.95
    h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
    result.writerow({"nb run": mean(life_time), "lifetime": h})

# 加载模型
model = CRFormer()
model.load_state_dict(torch.load('crformer_750_best.pth'))
model.eval()

# 创建你的终极优化器
optimizer = CRFormerOptimizer(model)

# 运行仿真
net = create_net()
net.simulate(optimizer=optimizer)  # 完美！