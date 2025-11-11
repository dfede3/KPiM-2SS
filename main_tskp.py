from joblib import Parallel, delayed
import numpy as np

from solver import Solver

# specific for the two-stage stochastic knapsack problem with setup
# load environment:
from environment import tskp_Env as Env 
# load problem-specific functions:
import pf_tskp

# list of solvers we test:
solvers = ['complete_enum','PiM','miqp']

# parameters of the tests
num_instances = 5
tmax = 30*60 
tol = 1e-5

r = 5 # number of first stage variables (n_x)

seed = 14

print('Tests for tskp:')

for N in [10, 15] : # number of second stage variables (n_y)
    for l in [10, 15, 20]:
        
        # create the instances 
        env_list = [Env(N=N, l=l, r=r, inst_num=i) for i in range(1, num_instances+1)]
        Parallel(n_jobs=-1)(delayed(env.make_test_inst)() for env in env_list)

        for K in [2, 3, 4, 5]:
            assert not (np.mod(N, r))
            print(f'Size {N} with {l} scenarios, K = {K}')
            pp = pf_tskp.tskp_Inst(tmax = tmax)

            for env in env_list :
                S = Solver(problem = pp, env=env, K = K, max_time = tmax, tol = tol, print_info = False)
                S.test_problem(solvers, seed = seed) 