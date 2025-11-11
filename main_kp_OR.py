from joblib import Parallel, delayed
from solver import Solver

# specific for the single-stage stochastic knapsack problem
# load environment:
from environment import kp_Env as Env
# load problem-specific functions:
import pf_kp_OR

# list of solvers we test:
solvers = ['complete_enum','PiM','miqp']

# parameters of the tests
num_instances = 5
tmax = 30*60 
tol = 1e-5

seed = 14

print('Tests for kp:')

for N in [10]: # number of second stage variables (n_y)
    for l in [10, 15, 20]:
        
        # create the instances 
        env_list = [Env(N=N, l=l, inst_num=i) for i in range(1, num_instances+1)]
        Parallel(n_jobs=-1)(delayed(env.make_test_inst)() for env in env_list)

        for K in [2, 3, 4, 5]:
            print(f'Size {N} with {l} scenarios, K = {K}')
            pp = pf_kp_OR.kp_Inst(tmax = tmax)

            for env in env_list :
                S = Solver(problem = pp, env=env, K = K, max_time = tmax, tol = tol, print_info = False)
                S.test_problem(solvers, seed) 