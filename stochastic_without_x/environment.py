import numpy as np
import copy
import os
import matplotlib.pyplot as plt

class kp_Env:
    def __init__(self,N,l,inst_num=0) :
        self.inst_num = inst_num

        self.N = N # number of y variables
        self.N_scen = l # number of scenarios

    def make_test_inst(self, save_env=True):
        
        self.prob = np.ones(self.N_scen)/self.N_scen # uniform probability
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

        env = copy.deepcopy(self) 
        env.weight = {}
        env.cost = {}
        env.W = {}
        env.capacity = {}
        env.min_capacity = {}
        rng = np.random.RandomState(env.inst_num) 
        for s in range(self.N_scen) :
            env.weight[s] = np.round(rng.uniform(0.0, 1.0, self.N), 3)
            env.cost[s]   = np.round(rng.uniform(0.0, 1.0, self.N), 3)
            env.W[s] = round(np.sum(env.weight[s]), 3)
            env.capacity[s] = round(0.75 * env.W[s], 3)
            env.min_capacity[s] = round(0.95 * env.capacity[s], 3)

        if save_env:
            env.write_test_inst() 
    def read_test_inst(self):
        inst_path = f"data/kp/kp_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.cost = {}
        for s in range(self.N_scen) : 
            self.cost[s] = np.array(list(map(lambda x: float(x), f_lines[1+s].replace('\n', '').split(' '))))
        self.weight = {}
        for s in range(self.N_scen) : 
            self.weight[s] = np.array(list(map(lambda x: float(x), f_lines[1+self.N_scen+s].replace('\n', '').split(' '))))
        self.prob = np.array(list(map(lambda x: float(x), f_lines[1+ 2*self.N_scen].replace('\n', '').split(' '))))
        self.W = np.array(list(map(lambda x: float(x), f_lines[2+ 2*self.N_scen].replace('\n', '').split(' '))))
        self.capacity = np.array(list(map(lambda x: float(x), f_lines[3+ 2*self.N_scen].replace('\n', '').split(' '))))
        self.min_capacity = np.array(list(map(lambda x: float(x), f_lines[4+ 2*self.N_scen].replace('\n', '').split(' '))))

        self.upper_bound = np.inf
        self.lower_bound = - np.inf

    def write_test_inst(self):
        os.makedirs("data/kp", exist_ok=True) 
        inst_path = f"data/kp/kp_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + "\n")
        for s in range(self.N_scen) : 
            f.write(" ".join(self.cost[s].astype(str)) + "\n")
        for s in range(self.N_scen) : 
            f.write(" ".join(self.weight[s].astype(str)) + "\n")
        f.write(" ".join(self.prob.astype(str)) + "\n")
        f.write(" ".join(str(v) for v in self.W.values()))
        f.write("\n")
        f.write(" ".join(str(v) for v in self.capacity.values()))
        f.write("\n")
        f.write(" ".join(str(v) for v in self.min_capacity.values()))
        f.write("\n")
        f.close()