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

class tskp_Env:
    def __init__(self, N, l, r, inst_num=0) :
        self.inst_num = inst_num

        self.r = r # number of classes
        self.N = N
        self.q = int(self.N / self.r)
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
        for s in range(env.N_scen) :
            env.weight[s] = np.round(rng.uniform(10.0, 100.0, self.N), 3)
            env.cost[s] = np.round(rng.uniform(0.0, 100.0, self.N), 3)
            env.W[s] = round(np.sum(env.weight[s]), 3)
            env.capacity[s] = round(0.5 * env.W[s], 3)
            env.min_capacity[s] = round(0.95 * env.capacity[s], 3)

        env.c = {}
        env.d = {}
        for h in range(env.r):
            env.c[h] = round(0.20 * sum( sum(env.cost[s][i] for i in range(h*env.q,(h+1)*env.q)) for s in range(env.N_scen)) / env.N_scen, 3)
            env.d[h] = round(0.20 * sum( sum(env.weight[s][i] for i in range(h*env.q,(h+1)*env.q)) for s in range(env.N_scen)) / env.N_scen, 3)

        if save_env:
            env.write_test_inst() 

    def read_test_inst(self):
        inst_path = f"data/tskp/tskp_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.r = int(inst_info[2])
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
        self.c = np.array(list(map(lambda x: float(x), f_lines[5+ 2*self.N_scen].replace('\n', '').split(' '))))
        self.d = np.array(list(map(lambda x: float(x), f_lines[6+ 2*self.N_scen].replace('\n', '').split(' '))))

        self.q = int(self.N/self.r)
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

    def write_test_inst(self):
        os.makedirs("data/tskp", exist_ok=True) 
        inst_path = f"data/tskp/tskp_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + " ")
        f.write(str(self.r) + "\n")
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
        f.write(" ".join(str(v) for v in self.c.values()))
        f.write("\n")
        f.write(" ".join(str(v) for v in self.d.values()))
        f.write("\n")

        f.close()

class cfl_Env:
    def __init__(self,N,l,r,inst_num=0) :
        self.inst_num = inst_num
    
        self.r = r # number of first stage var
        self.N = N # number of second stage var

        self.N_scen = l # number of scenarios 

    def make_test_inst(self, save_env=True):
        
        self.prob = np.ones(self.N_scen)/self.N_scen # uniform probability
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

        env = copy.deepcopy(self) 
        env.cost1 = {} 
        env.cost2 = {}
        env.V   = {} 
        env.dem = {}
        rng = np.random.RandomState(env.inst_num)
        env.cost1 = np.round(rng.uniform(100.0, 1000.0, self.r))
        env.V = np.round(rng.uniform(50,500, self.r))
        for s in range(env.N_scen) :
            for i in range(env.r) :
                env.cost2[s,i] = np.round(rng.uniform(0, 100, self.N))
                env.dem[s,i]   = np.round(rng.uniform(1, 100, self.N))

        if save_env:
            env.write_test_inst()

    def read_test_inst(self):
        inst_path = f"data/cfl/cfl_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.r = int(inst_info[2])
        self.cost1 = np.array(list(map(lambda x: float(x), f_lines[1].replace('\n', '').split(' '))))
        self.V = np.array(list(map(lambda x: float(x), f_lines[2].replace('\n', '').split(' '))))
        self.cost2 = {}
        self.dem = {}
        for s in range(self.N_scen) :
            for i in range(self.r) : 
                self.cost2[s,i] = np.array(list(map(lambda x: float(x), f_lines[3+s*self.r+i].replace('\n', '').split(' '))))
        for s in range(self.N_scen) :
            for i in range(self.r) : 
                self.dem[s,i] = np.array(list(map(lambda x: float(x), f_lines[3+self.N_scen*self.r + s*self.r+i].replace('\n', '').split(' '))))
        
        self.prob = np.array(list(map(lambda x: float(x), f_lines[3+ 2*(self.N_scen*self.r)].replace('\n', '').split(' '))))
        
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

    def write_test_inst(self):
        os.makedirs("data/cfl", exist_ok=True) 
        inst_path = f"data/cfl/cfl_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + " ")
        f.write(str(self.r) + "\n")
        f.write(" ".join(str(v) for v in self.cost1))
        f.write("\n")
        f.write(" ".join(str(v) for v in self.V))
        f.write("\n")
        for s in range(self.N_scen) :
            for i in range(self.r): 
                f.write(" ".join(self.cost2[s,i].astype(str)) + "\n")
        for s in range(self.N_scen) :
            for i in range(self.r):
                f.write(" ".join(self.dem[s,i].astype(str)) + "\n")
        f.write(" ".join(self.prob.astype(str)) + "\n")
        f.write("\n")

        f.close()

class qtskp_Env:
    def __init__(self,N,l,r,inst_num=0) :
        self.inst_num = inst_num
    
        self.r = r # number of classes
        self.N = N
        self.q = int(self.N / self.r)
        self.N_scen = l # number of scenarios 

    def make_test_inst(self, save_env=True):
        
        self.prob = np.ones(self.N_scen)/self.N_scen 
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

        env = copy.deepcopy(self) 
        env.weight = {}
        env.cost = {}
        env.qcost = {}
        env.W = {}
        env.capacity = {}
        env.min_capacity = {}
        rng = np.random.RandomState(env.inst_num) 
        for s in range(env.N_scen) :
            env.weight[s] = np.round(rng.uniform(10.0, 100.0, self.N), 3)
            env.cost[s] = np.round(rng.uniform(0.0, 100.0, self.N), 3)
            for i in range(self.N):
                env.qcost[s,i] = np.round(rng.uniform(0.0, 100.0, self.N), 3)
            env.W[s] = round(np.sum(env.weight[s]), 3)
            env.capacity[s] = round(0.5 * env.W[s], 3)
            env.min_capacity[s] = round(0.95 * env.capacity[s], 3)

        env.c = {}
        env.d = {}
        for h in range(env.r):
            env.c[h] = round(0.20 * sum( sum(env.cost[s][i] for i in range(h*env.q,(h+1)*env.q)) for s in range(env.N_scen)) / env.N_scen, 3)
            env.d[h] = round(0.20 * sum( sum(env.weight[s][i] for i in range(h*env.q,(h+1)*env.q)) for s in range(env.N_scen)) / env.N_scen, 3)

        if save_env:
            env.write_test_inst() 

    def read_test_inst(self):
        inst_path = f"data/qtskp/qtskp_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.r = int(inst_info[2])
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
        self.c = np.array(list(map(lambda x: float(x), f_lines[5+ 2*self.N_scen].replace('\n', '').split(' '))))
        self.d = np.array(list(map(lambda x: float(x), f_lines[6+ 2*self.N_scen].replace('\n', '').split(' '))))

        self.qcost = {}
        for s in range(self.N_scen) : 
            for i in range(self.N) :
                self.qcost[s,i] = np.array(list(map(lambda x: float(x), f_lines[7+ 2*self.N_scen +s].replace('\n', '').split(' '))))

        self.q = int(self.N/self.r)
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

    def write_test_inst(self):
        os.makedirs("data/qtskp", exist_ok=True)
        inst_path = f"data/qtskp/qtskp_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + " ")
        f.write(str(self.r) + "\n")
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
        f.write(" ".join(str(v) for v in self.c.values()))
        f.write("\n")
        f.write(" ".join(str(v) for v in self.d.values()))
        f.write("\n")
        for s in range(self.N_scen) :
            for i in range(self.N) :
                f.write(" ".join(self.qcost[s,i].astype(str)) + "\n")

        f.close()

class qcfl_Env:
    def __init__(self,N,l,r,inst_num=0) :
        self.inst_num = inst_num
    
        self.r = r # number of first stage var
        self.N = N # number of second stage var

        self.N_scen = l # number of scenarios
        self.h = 25

    def make_test_inst(self, save_env=True):
        
        self.prob = np.ones(self.N_scen)/self.N_scen # uniform probability
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

        env = copy.deepcopy(self) 
        env.cost1 = {} 
        env.cost2 = {}
        env.V   = {}
        env.dem = {}
        rng = np.random.RandomState(env.inst_num)
        env.cost1 = np.round(rng.uniform(100.0, 1000.0, self.r))
        env.V = np.round(rng.uniform(50,500, self.r))
        for s in range(env.N_scen) :
            for i in range(env.r) :
                env.cost2[s,i] = np.round(rng.uniform(0, 100, self.N))
                env.dem[s,i]   = np.round(rng.uniform(1, 100, self.N))

        if save_env:
            env.write_test_inst() 

    def read_test_inst(self):
        inst_path = f"data/qcfl/qcfl_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"

        with open(inst_path, 'r') as f:
            f_lines = f.readlines()

        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        self.inst_num = int(inst_info[0])
        self.N = int(inst_info[1])
        self.r = int(inst_info[2])
        self.h = int(inst_info[3])
        self.cost1 = np.array(list(map(lambda x: float(x), f_lines[1].replace('\n', '').split(' '))))
        self.V = np.array(list(map(lambda x: float(x), f_lines[2].replace('\n', '').split(' '))))
        self.cost2 = {}
        self.dem = {}
        for s in range(self.N_scen) :
            for i in range(self.r) : 
                self.cost2[s,i] = np.array(list(map(lambda x: float(x), f_lines[3+s*self.r+i].replace('\n', '').split(' '))))
        for s in range(self.N_scen) :
            for i in range(self.r) : 
                self.dem[s,i] = np.array(list(map(lambda x: float(x), f_lines[3+self.N_scen*self.r + s*self.r+i].replace('\n', '').split(' '))))
        
        self.prob = np.array(list(map(lambda x: float(x), f_lines[3+ 2*(self.N_scen*self.r)].replace('\n', '').split(' '))))
        
        self.upper_bound = np.inf
        self.lower_bound = - np.inf

    def write_test_inst(self):
        os.makedirs("data/qcfl", exist_ok=True) 
        inst_path = f"data/qcfl/qcfl_env_N{self.N}_" \
                    f"l{self.N_scen}_" \
                    f"s{self.inst_num}.txt"
        f = open(inst_path, "w+")
        f.write(str(self.inst_num) + " ")
        f.write(str(self.N) + " ")
        f.write(str(self.r) + " ")
        f.write(str(self.h) + "\n")
        f.write(" ".join(str(v) for v in self.cost1))
        f.write("\n")
        f.write(" ".join(str(v) for v in self.V))
        f.write("\n")
        for s in range(self.N_scen) :
            for i in range(self.r): 
                f.write(" ".join(self.cost2[s,i].astype(str)) + "\n")
        for s in range(self.N_scen) :
            for i in range(self.r):
                f.write(" ".join(self.dem[s,i].astype(str)) + "\n")
        f.write(" ".join(self.prob.astype(str)) + "\n")
        f.write("\n")

        f.close()