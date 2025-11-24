import math
from scipy.spatial import distance
from config import params


class Node:
    def __init__(self, location=None, com_ran=None, energy=None, energy_max=None,
                 id=None, energy_thresh=None, prob=params.PROB, is_active=True):
        self.location = location
        self.com_ran = com_ran
        self.energy = energy
        self.energy_max = energy_max
        self.energy_thresh = energy_thresh
        self.prob = prob
        self.id = id
        self.is_active = is_active

        # 网络属性
        self.neighbor = []
        self.level = 0
        self.cluster_id = -1
        self.load = 0.015
        self.is_request = False

        # 统计信息
        self.used_energy = 0.0
        self.check_point = [{"E_current": self.energy, "time": 0, "avg_e": 0.0}]

    def charge(self, mc):
        """充电函数"""
        dist = distance.euclidean(self.location, mc.current)
        if dist > self.com_ran:
            return 0
        p = 80 / (dist + 10) ** 1.5
        return min(p, 100)

    def send(self, net, package):
        """发送数据包"""
        d0 = math.sqrt(params.EFS / params.EMP)
        package.update_path(self.id)

        if distance.euclidean(self.location, params.BASE_STATION) > self.com_ran:
            receiver_id = self._find_receiver(net)
            if receiver_id != -1:
                d = distance.euclidean(self.location, net.nodes[receiver_id].location)
                e_send = params.ET + params.EFS * d ** 2 if d <= d0 else params.ET + params.EMP * d ** 4
                self._consume_energy(e_send * package.size)
                net.nodes[receiver_id].receive(package)
                net.nodes[receiver_id].send(net, package)
        else:
            package.is_success = True
            d = distance.euclidean(self.location, params.BASE_STATION)
            e_send = params.ET + params.EFS * d ** 2 if d <= d0 else params.ET + params.EMP * d ** 4
            self._consume_energy(e_send * package.size)
            package.update_path(-1)

        self._check_active(net)

    def receive(self, package):
        """接收数据包"""
        self._consume_energy(params.ER * package.size)

    def _find_receiver(self, net):
        """寻找下一跳接收节点"""
        if not self.is_active:
            return -1

        candidate = [neighbor_id for neighbor_id in self.neighbor
                     if net.nodes[neighbor_id].level < self.level and net.nodes[neighbor_id].is_active]

        if candidate:
            distances = [distance.euclidean(net.nodes[nid].location, params.BASE_STATION) for nid in candidate]
            return candidate[np.argmin(distances)]
        return -1

    def _consume_energy(self, amount):
        """消耗能量"""
        self.energy -= amount
        self.used_energy += amount

    def _check_active(self, net):
        """检查节点是否存活"""
        if self.energy < 0 or not self.neighbor:
            self.is_active = False
        else:
            active_neighbors = sum(1 for nid in self.neighbor if net.nodes[nid].is_active)
            self.is_active = active_neighbors > 0

    def request_charge(self, mc, current_time):
        """请求充电"""
        if not self.is_request:
            mc.add_request({
                "id": self.id,
                "energy": self.energy,
                "time": current_time
            })
            self.is_request = True