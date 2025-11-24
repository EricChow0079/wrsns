# 网络参数
BASE_STATION = (500.0, 500.0)
DEPOT = (0.0, 0.0)

# 能量参数
ALPHA = 20.0
BETA = 20.0
E_MC_THRESH = 10

# 通信参数
PROB = 1.0
B = 200.0  # 数据包大小
B_ENERGY = 0.0  # 能量信息包大小

# 能量消耗参数
ER = 0.0000001    # 接收能耗
ET = 0.00000005   # 发送能耗
EFS = 0.00000000001  # 自由空间能耗系数
EMP = 0.0000000000000013  # 多路径衰减能耗系数

# CR-Former参数
CRF_MODEL_PATH = 'model/crformer_750_best.pth'
RISK_THRESHOLD = 0.6
RESCUE_COOLDOWN = 10
CLUSTER_NUM = 15

# 仿真参数
MAX_TIME_STEPS = 10000
MIN_ALIVE_NODES = 30
CASCADE_LOAD_FACTOR = 1.5