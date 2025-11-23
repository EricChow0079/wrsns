import math
from scipy.spatial import distance

import Parameter as para
from Node_Method import to_string, find_receiver, request_function, estimate_average_energy


class Node:
    def __init__(self, location=None, com_ran=None, sen_ran=None, energy=None, prob=para.prob, avg_energy=0.0,
                 len_cp=10, id=None, is_active=True, energy_max=None, energy_thresh=None,):
        self.location = location  # location of sensor
        self.com_ran = com_ran  # communication range
        self.sen_ran = sen_ran  # sensing range
        self.energy = energy  # energy of sensor
        self.energy_max = energy_max  # capacity of sensor
        self.energy_thresh = energy_thresh  # threshold to sensor send request for mc
        self.prob = prob  # probability of sending data
        self.check_point = [{"E_current": self.energy, "time": 0, "avg_e": 0.0}]  # check point of information of sensor
        self.used_energy = 0.0  # energy was used from last check point to now
        self.avg_energy = avg_energy  # average energy of sensor
        self.len_cp = len_cp  # length of check point list
        self.id = id  # identify of sensor
        self.neighbor = []  # neighborhood of sensor
        self.is_active = is_active  # statement of sensor. If sensor dead, state is False
        self.is_request = False
        self.level = 0
        self.cluster_id = -1  # 所属簇ID，-1表示未分配
        self.load = 0.015  # 基础数据产生负载（单位：J/step），可随机
        self.transfer_load_when_die = True  # 死时是否转移负载

    def set_average_energy(self, func=estimate_average_energy):
        """
        calculate average energy of sensor
        :param func: function to calculate
        :return: set value for average energy with estimate function is func
        """
        self.avg_energy = func(self)

    def set_check_point(self, t):
        """
        add new check point in check_point list
        :param t: time stem
        :return: if queue of check point is not full, add new check point
        """
        if len(self.check_point) >= self.len_cp:
            self.check_point.pop(0)
        self.check_point.append(
            {"E_current": self.energy, "time": t, "avg_e": self.used_energy / (t - self.check_point[-1]["time"])})
        self.avg_energy = self.check_point[-1]["avg_e"]
        self.used_energy = 0.0

    def charge(self, mc):
        dist = distance.euclidean(self.location, mc.current)
        if dist > self.com_ran:
            return 0
        # 终极加强版：功率翻倍！
        p = 80 / (dist + 10) ** 1.5  # 直接硬编码高功率（原来可能只有几瓦）
        return min(p, 100)  # 最大100J/s（超级快！）

    def send(self, net=None, package=None, receiver=find_receiver, is_energy_info=False):
        """
        send package
        :param package:
        :param net: the network
        :param receiver: the function calculate receiver node
        :param is_energy_info: if this package is energy package, is_energy_info will be true
        :return: send package to the next node and reduce energy of this node
        """
        d0 = math.sqrt(para.EFS / para.EMP)
        package.update_path(self.id)
        if distance.euclidean(self.location, para.base) > self.com_ran:
            receiver_id = receiver(self, net)
            if receiver_id != -1:
                d = distance.euclidean(self.location, net.node[receiver_id].location)
                e_send = para.ET + para.EFS * d ** 2 if d <= d0 else para.ET + para.EMP * d ** 4
                self.energy -= e_send * package.size
                self.used_energy += e_send * package.size
                net.node[receiver_id].receive(package)
                net.node[receiver_id].send(net, package, receiver, is_energy_info)
        else:
            package.is_success = True
            d = distance.euclidean(self.location, para.base)
            e_send = para.ET + para.EFS * d ** 2 if d <= d0 else para.ET + para.EMP * d ** 4
            self.energy -= e_send * package.size
            self.used_energy += e_send * package.size
            package.update_path(-1)
        self.check_active(net)

    def receive(self, package):
        """
        receive package from other node
        :param package: size of package
        :return: reduce energy of this node
        """
        self.energy -= para.ER * package.size
        self.used_energy += para.ER * package.size

    def check_active(self, net):
        """
        check if the node is alive
        :param net: the network
        :return: None
        """
        if self.energy < 0 or len(self.neighbor) == 0:
            self.is_active = False
        else:
            a = [1 for neighbor in self.neighbor if net.node[neighbor].is_active]
            self.is_active = True if len(a) > 0 else False

    def request(self, mc, t, request_func=request_function):
        """
        send a message to mc if the energy is below a threshold
        :param mc: mobile charger
        :param t: time to send request
        :param request_func: structure of message
        :return: None
        """
        self.set_check_point(t)
        # print(self.check_point)
        if not self.is_request:
            request_func(self, mc, t)
            self.is_request = True

    def print_node(self, func=to_string):
        """
        print node information
        :param func: print function
        :return:
        """
        func(self)
