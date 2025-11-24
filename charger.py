from scipy.spatial import distance
from config import params


class MobileCharger:
    def __init__(self, energy=108000, capacity=108000, e_move=0.2,
                 e_self_charge=0.1, velocity=30, start=params.DEPOT):
        self.energy = energy
        self.capacity = capacity
        self.e_move = e_move
        self.e_self_charge = e_self_charge
        self.velocity = velocity

        # 位置状态
        self.current = start
        self.start = start
        self.end = start

        # 运行状态
        self.is_stand = False
        self.is_self_charge = False
        self.is_active = False
        self.end_time = -1

        # 请求队列
        self.requests = []

    def add_request(self, request):
        """添加充电请求"""
        self.requests.append(request)

    def update_location(self):
        """更新位置"""
        if distance.euclidean(self.current, self.end) < 1e-3:
            self.current = self.end
            return

        dx = (self.end[0] - self.start[0])
        dy = (self.end[1] - self.start[1])
        dist_total = distance.euclidean(self.start, self.end)

        if dist_total == 0:
            return

        x_hat = self.current[0] + dx / dist_total * self.velocity
        y_hat = self.current[1] + dy / dist_total * self.velocity

        # 检查是否到达终点
        if (self.end[0] - self.current[0]) * (self.end[0] - x_hat) <= 0:
            self.current = self.end
        else:
            self.current = (x_hat, y_hat)

        self.energy -= self.e_move

    def charge_nodes(self, network):
        """为节点充电"""
        for node in network.nodes:
            if node.is_active and distance.euclidean(self.current, node.location) <= node.com_ran:
                charge_power = node.charge(self)
                self.energy -= charge_power

    def self_charge(self):
        """自我充电"""
        self.energy = min(self.energy + self.e_self_charge, self.capacity)

    def update_state(self):
        """更新状态"""
        self.is_stand = distance.euclidean(self.current, self.end) < 1
        self.is_self_charge = distance.euclidean(self.current, params.DEPOT) < 1e-3

    def run(self, network, current_time, optimizer=None):
        """运行充电车"""
        if (not self.is_active and self.requests) or abs(current_time - self.end_time) < 1:
            self.is_active = True
            # 过滤无效请求
            self.requests = [req for req in self.requests
                             if network.nodes[req["id"]].is_active]

            if not self.requests:
                self.is_active = False
                return

            # 获取下一个目标
            if optimizer:
                next_location, charge_time = optimizer.get_next_target(network)
                self.start = self.current
                self.end = next_location
                move_time = distance.euclidean(self.start, self.end) / self.velocity
                self.end_time = current_time + move_time + charge_time

        elif self.is_active:
            if not self.is_stand:
                self.update_location()
            elif not self.is_self_charge:
                self.charge_nodes(network)
            else:
                self.self_charge()

        # 低电量返回基站
        if (self.energy < params.E_MC_THRESH and not self.is_self_charge
                and self.end != params.DEPOT):
            self.return_to_depot(current_time)

        self.update_state()

    def return_to_depot(self, current_time):
        """返回基站充电"""
        self.start = self.current
        self.end = params.DEPOT
        self.is_stand = False
        charge_time = self.capacity / self.e_self_charge
        move_time = distance.euclidean(self.start, self.end) / self.velocity
        self.end_time = current_time + move_time + charge_time