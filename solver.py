from datetime import datetime
import gurobipy as gp
import numpy as np
import random
import copy
from sympy.functions.combinatorial.numbers import stirling

import time
from tabulate import tabulate
import psutil

class Solver :
    def __init__(self, problem, method = 'bnb', env = None, tol = 1e-5, ram = 95, K = 2,
                 max_time = 10*60, mTL = 1, print_info = False):
        self.problem = problem 
        self.env = env
        self.tol = tol
        self.ram = ram
        self.K   = K
        self.max_time   = max_time
        self.mTL = mTL
        self.print_info = print_info
        self.stirling_n_K_2 = stirling(self.env.N_scen, self.K, kind =2)
        self.env.read_test_inst()
    
    def set_sol_method(self, method):
        self.method = method
    
    def set_seed(self, seed):
        if seed is not None : 
            self.rng = np.random.RandomState(seed)
        else: 
            print("Error : seed not defined")

    def solve(self, method = None, seed = None): 
        if method is not None : 
            self.set_sol_method(method)
        if seed is not None :
            self.set_seed(seed)
        else :
            print("Error : seed not defined")

        if self.method   == 'PiM':
            sol, info, ram_issue = self.solveBnB(n_sel = 'breadth', s_sel = 'sfeas', fixx = True)
        elif self.method == 'miqp':
            sol, info, ram_issue = self.solveMiqp()
        elif self.method == 'complete_enum':
            sol, info, ram_issue = self.solveCompleteEnum()
        else :
            raise NotImplementedError('SOLVER UNKNOWN')

        return sol, info, ram_issue
    
    def test_problem(self, solvers, seed) : 
        res_tab  = []
        inc_UB_t = {}

        for solver in solvers :
            _, info, ram_issue = self.solve(method = solver, seed = seed)
            print(f'Solver {solver} ended!')

            if ram_issue :
                print('Warning: Ram Issue!')
                return

            n_leaves = stirling(self.env.N_scen, self.K, kind =2)
            p_leaves = info['tot_leaves']/n_leaves * 100

            if info['inc_ub_t'] is not None :
                inc_UB_t[solver] = info['inc_ub_t']
            
            res_tab.append([solver, info['time'], np.round(info['f_kopt'],3), info['n_prunes'], info['prune_infeas_count'], info['tot_nodes'], info['tot_leaves'], p_leaves])

        table = tabulate(res_tab, headers = ['Algorithm', 't', 'f_kopt', 'n_prunes', 'n_feas_prunes', 'n_nodes', 'n_leaves', '% leaves'], tablefmt='orgtbl')
        print(table)
        
        return inc_UB_t

    ########################################################################################################

    def solveBnB(self, empty = False, n_sel = None, s_sel = 'smax', fixx = False, lazy = True) :
        # 0) importing the environment 
        env = copy.deepcopy(self.env)
        rng = self.rng
        gp_env = gp.Env()

        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("Threads", 1)

        # ------------------------------------------------------------------
        # 1) initialization of parameters / counters 
        inc_x = {} ; inc_y = {} ; inc_UB_t = {}
        x = fx = fy = fys =  None
        true_lb = None

        N_set = [] # list of the nodes of the bnb-tree

        prune_count = prune_infeas_count = 0 # counters for prunings
        tot_nodes   = tot_leaves = 0 # counters for nodes / leaves that are visited during the search
        tot_solves  = 0 # number of models that were (actually) optimized
        mp_time     = sp_time = 0    # time spent solving an opt problem / time spent finding induced partitions 
        resolve_times = solved_fixing_vars = 0
        new_model_times = 0
        lb_useful = 0
        ngc_count = 0 # number of lazy constratins
        ngc_list  = [] # list of lazy constraints
        TL_reached = 0
        mTL = self.mTL # time limit for the problem at a node

        skip    = False # True if the opt sol of the current node is already known
        clear_f = False # True if we must restore the previous version of fy = None

        now = datetime.now().time()
        print(f"Instance {env.inst_num}: bnb-like method started at {now}")
        start_time = time.time()

        # ------------------------------------------------------------------
        # 2) initialization of upper and lower bounds
        UB_i, x_i, y_i = (env.upper_bound,[],[])
        tau_i = None
        lb = env.lower_bound
        
        # ------------------------------------------------------------------
        # 3) exploration of the root node
        s, k_new  = None, None  # s : scenario to generate child nodes /
                                # k_new : index of the subset containing s in the child-node that is visited first
        s_feas, s_new = None, None
        new_model   = True
        prune_state = False

        tau = {k: [] for k in range(self.K)} # root node : all empty "partition"
        unseen_scen = [s for s in range(env.N_scen)] # no scenarios are in the current partition
        # counters update
        tot_nodes       += 1 
        tot_solves      += 1 
        new_model_times += 1

        # solving spip(tau) : it is the extensive formulation of the 2ss problem
        start_mp = time.time()
        lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_build(env, gp_env, self.K, tau, TL = mTL)
        mp_time += time.time() - start_mp

        if status == 'time_limit':
            TL_reached += 1
            skip = False
        elif status == 'infeasible':
            print('the original 2ss problem is infeasible') 
            # the original stochastic problem may itself be infeasible, in which case the K-adaptability problem is infeasible       
            return tau_i, {"time": time.time() - start_time, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "tot_solves": 0, 'inc_ub_t': inc_UB_t}, psutil.virtual_memory().percent > self.ram
        else : # status == 'optimal'
            true_lb = lb
            skip = True # we do not need to solve the following node
            # we do this even if fixx = False
            fx  = copy.deepcopy(x)
            fy  = copy.deepcopy(y)
            fys = copy.deepcopy(ys)

        # ------------------------------------------------------------------
        # 4) exploration of the first non-empty node
        tot_nodes += 1
        k_new = 0
        if s_new is not None:
            s = s_new
        else : 
            s = int(rng.choice(unseen_scen))

        tau[k_new].append(s)
        unseen_scen.remove(s)
        j = 1 # depth of the node we are at == number of scenarios contained in the partition
        u = env.N_scen - j # number of scenarios that are not contained in the current tau

        full_list = [k for k in range(self.K) if len(tau[k]) > 0] # here full_list = [0]
        M = len(full_list) # number of non-empty sets / 2nd stage strategies actually computed - here M = 1

        start_mp = time.time()
        lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_update(env, model, self.K, tau, k_new, s, fx, fy, fys, skip = skip, TL = mTL)
        mp_time += time.time() - start_mp
        
        if status == 'time_limit':
            TL_reached += 1
            true_lb = None
            fx  = None
            fy  = None
            fys = None
        elif status == 'optimal':
            true_lb = lb
            if fixx :
                fx  = copy.deepcopy(x) 
                fy  = copy.deepcopy(y)
                fys = copy.deepcopy(ys) 
            else :
                fx  = None 
                fy  = None
                fys = None

            # 4.1) we look for the partition induced by the solution (x,y)
            start_sp = time.time()
            feas, partition, ub_tau, ngc_count, model, ngc_list, s_feas = self.induced_partition(env, model, tau, x, y, obj_R, unseen_scen, M, ngc_count, lazy, ngc_list)
            sp_time += time.time() - start_sp
            
            if feas : # i.e. partition contains the partition induced by tau
                status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(ub_tau, x, y, partition, start_time, 
                                                                                                    UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                if status_update : # if we successfully updated the incumbent upper bound, we check if we can now prune by bound 
                    prune_count, prune_state = self.check_prune_by_bound(true_lb, UB_i, prune_count, tot_nodes)
                    if prune_state : # if we can prune then we stop here 
                        return tau_i, {"time": time.time() - start_time, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram

        if M < self.K and u == self.K - M:
            tot_nodes += 1
            for k in range(u):
                tau[M+k].append(unseen_scen[k])

            fx  = None
            fy  = None
            fys = None
            tot_leaves += 1
            tot_solves += 1
            TL = self.max_time
            start_mp = time.time()
            lb, x, y,_, model, obj_R, _,status = self.problem.spip_build(env, gp_env, self.K, tau, TL = mTL)
            mp_time += time.time() - start_mp

            if status == 'time_limit':
                TL_reached += 1
                print('time limit reached')
                return tau_i, {"time": time.time() - start_time, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "tot_solves": tot_solves, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram
            elif status == 'optimal':
                status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(lb, x, y, tau, start_time, 
                                                                                        UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                prune_count += 1
            else :
                prune_count, prune_infeas_count, new_model = self.prune_by_infeas(prune_count, prune_infeas_count, tot_nodes)

            k_new     = None
            new_model = True
            model.dispose()
            return tau_i, {"time": time.time() - start_time, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram


        # 4.2) generating child nodes
        s = self.scen_selection(rng, status, s_sel, unseen_scen, s_new, s_feas)
        j += 1 # depth of the child nodes
        u -= 1 # number of scenarios that are not contained in the child nodes of tau
        N_set, k_new, s, skip = self.child_generation(M, s, rng, N_set, tau, fixx, fx, fy, fys, lb, j, y, status)
        new_model = False # depth first search
        # ------------------------------------------------------------------

        while (not(N_set == []) or (k_new is not None)) and time.time() - start_time < self.max_time:
            if new_model : 
                # search-strategy (node-selection strategy)
                if n_sel == 'breadth' :
                    tau_all = N_set.pop(0)
                elif n_sel == 'depth' :
                    tau_all = N_set.pop()
                else : # random selection of the new node to be visited :
                    new_pass = rng.randint(len(N_set))
                    tau_all = N_set.pop(new_pass)

                tau   = tau_all['tau'] 
                s     = tau_all['snew'] 
                k_new = tau_all['knew']
                if fixx :
                    fx = tau_all['fx']
                    fy = tau_all['fy']
                    fys = tau_all['fys']
                true_lb = tau_all['lb'] 
                j  = tau_all['j']
                u  = env.N_scen - j
                M = len([k for k in range(self.K) if len(tau[k]) > 0])
            elif skip : 
                # we build the new node
                tot_nodes += 1
                if M < self.K and u+1 == self.K - M:
                    for k in range(self.K-M):
                        tau[M+k].append(unseen_scen[k])
                        y[M+k] = copy.deepcopy(ys[unseen_scen[k]])
                    
                    status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(true_lb, x, y, tau, start_time, 
                                                                                            UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                    prune_count += 1
                    k_new     = None
                    new_model = True
                    leaf      = False
                    model.dispose()
                    continue
                
                tau[k_new].append(s)
                unseen_scen.remove(s)
                if fy is None :
                    clear_f = True # after we are done we will restore fy = None
                    fx  = copy.deepcopy(x)
                    fy  = copy.deepcopy(y) 
                    fys = copy.deepcopy(ys)
                # we update the constraints and we change the value of y[k_new] to ys[s_new] - without reoptimizing the model
                start_mp = time.time()
                lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_update(env, model, self.K, tau, k_new, s, fx, fy, fys, skip = skip, TL = mTL)
                mp_time += time.time() - start_mp
                skip = False
                
                if clear_f :
                    fx  = None 
                    fy  = None
                    fys = None
                else : 
                    fx = copy.deepcopy(x)
                    fy = copy.deepcopy(y)
                    fys = copy.deepcopy(ys)

                full_list = [k for k in range(self.K) if len(tau[k]) > 0]
                M = len(full_list) 

                start_sp = time.time()
                feas, partition, ub_tau, ngc_count, model, ngc_list, s_feas = self.induced_partition(env, model, tau, x, y, obj_R, unseen_scen, M, ngc_count, lazy, ngc_list)
                sp_time += time.time() - start_sp

                if feas : 
                    status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(ub_tau, x, y, partition, start_time, 
                                                                                                        UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                    if status_update :
                        prune_count, prune_state = self.check_prune_by_bound(lb, UB_i, prune_count, tot_nodes)
                        if prune_state : 
                            new_model = True
                            k_new = None
                            model.dispose()
                            continue

                # we prepare to build the child nodes :
                j += 1
                u -= 1
                # scenario selection strategy : 
                s = self.scen_selection(rng, status, s_sel, unseen_scen, s_new, s_feas)

                if u == 0 :
                    leaf = True
                    fx  = None
                    fy  = None
                    fys = None
                    K_prime = min(self.K, full_list[-1] + 2)
                    for k_new in range(K_prime) :
                        tot_nodes  += 1
                        tot_leaves += 1
                        tot_solves += 1
                        tau[k_new].append(s)
                        TL = self.max_time
                        start_mp = time.time()
                        lb, x, y,_, model, obj_R, _,status = self.problem.spip_update(env, model, self.K, tau, k_new, s, fx, fy, fys, leaf=leaf, skip = False, TL=TL)
                        mp_time += time.time() - start_mp

                        if status == 'time_limit':
                            TL_reached += 1
                            print('time limit reached')
                            return tau_i, {"time": time.time() - start_time, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "tot_solves": tot_solves, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram
                        elif status == 'optimal':
                            status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(lb, x, y, tau, start_time, 
                                                                                                    UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                            prune_count += 1
                        else :
                            prune_count, prune_infeas_count, new_model = self.prune_by_infeas(prune_count, prune_infeas_count, tot_nodes)

                        tau[k_new].remove(s)

                    k_new     = None
                    new_model = True
                    leaf      = False
                    model.dispose()
                    continue
                else :
                    N_set, k_new, s, skip = self.child_generation(M, s, rng, N_set, tau, fixx, fx, fy, fys, true_lb, j, y, status)
                    new_model = False
                    continue

            # in true_lb we store the lb of the parent node of tau: we check if we can
            # prune by bound without solving the model corresponding to tau

            # true_lb is None if in the parent node we solved the model with fixed values of x and y,
            # i.e. it is not a "real" lower bound and we should not use it to prune

            tot_nodes += 1

            if true_lb is not None: 
                prune_count, prune_state = self.check_prune_by_bound(true_lb, UB_i, prune_count, tot_nodes)
                if prune_state : 
                    lb_useful += 1
                    new_model = True
                    k_new = None 
                    model.dispose()
                    continue
            
            tau[k_new].append(s)
            unseen_scen, _ = self.not_in_tau(env, tau)

            full_list = [k for k in range(self.K) if len(tau[k]) > 0]
            M = len(full_list) 

            if M < self.K and u == self.K - M:
                for k in range(u):
                    tau[M+k].append(unseen_scen[k])

                fx  = None
                fy  = None
                fys = None
                tot_leaves += 1
                tot_solves += 1
                TL = self.max_time
                start_mp = time.time()
                lb, x, y,_, model, obj_R, _,status = self.problem.spip_build(env, gp_env, self.K, tau, TL = mTL)
                mp_time += time.time() - start_mp

                if status == 'time_limit':
                    TL_reached += 1
                    print('time limit reached')
                    return tau_i, {"time": time.time() - start_time, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "tot_solves": tot_solves, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram
                elif status == 'optimal':
                    status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(lb, x, y, tau, start_time, 
                                                                                            UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                    prune_count += 1
                else :
                    prune_count, prune_infeas_count, new_model = self.prune_by_infeas(prune_count, prune_infeas_count, tot_nodes)

                k_new     = None
                new_model = True
                model.dispose()
                continue
            
            if true_lb is not None :
                true_lb = None

            if fixx and fy is not None : 
                solved_fixing_vars += 1
                if new_model :
                    tot_solves += 1
                    new_model_times += 1 
                    start_mp = time.time()
                    lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_build(env, gp_env, self.K, tau, k_new, fx, fy, fys, TL = mTL)
                    mp_time += time.time() - start_mp
                    if status == 'time_limit':
                        TL_reached += 1
                        fx  = None
                        fy  = None
                        fys = None
                else :
                    tot_solves += 1
                    start_mp = time.time()
                    lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_update(env, model, self.K, tau, k_new, s, fx, fy, fys, TL = mTL)
                    mp_time += time.time() - start_mp
                    if status == 'time_limit':
                        TL_reached += 1
                        fx  = None 
                        fy  = None
                        fys = None
                if status == 'infeasible': # it is infeasible but we were fixing the values of some of the variables : we rerun without fixing
                    tot_solves    += 1
                    resolve_times += 1 # we optimize the same model (same constraints and obj_fun) without fixing the values of x and y
                    start_mp = time.time()
                    lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_resolve(env, self.K, tau, model, TL = mTL)
                    mp_time += time.time() - start_mp
                    if status == 'time_limit':
                        TL_reached += 1
                        fx  = None 
                        fy  = None
                        fys = None
                    elif status == 'infeasible':
                        prune_count, prune_infeas_count, new_model = self.prune_by_infeas(prune_count, prune_infeas_count, tot_nodes)
                        k_new = None 
                        model.dispose()
                        continue
                    else : # status == 'optimal' of the new model
                        fx  = copy.deepcopy(x) 
                        fy  = copy.deepcopy(y)
                        fys = copy.deepcopy(ys)
                        true_lb = lb
                        prune_count, prune_state = self.check_prune_by_bound(true_lb, UB_i, prune_count, tot_nodes)
                        if prune_state :
                            new_model = True
                            k_new = None
                            model.dispose()
                            continue
                elif status == 'optimal':
                    fx  = copy.deepcopy(x)
                    fy  = copy.deepcopy(y)
                    fys =  copy.deepcopy(ys)
                    true_lb = None
                    # check prune by bound : (if it is satisfied we need to reoptimize) true_lb = None != lb
                    prune_count, prune_state = self.check_prune_by_bound(lb, UB_i, prune_count, tot_nodes)
                    if prune_state : # i.e. if we can prune by bound, then we must wait to prune and first re-optimize the model:
                        prune_count -= 1 # we do not prune yet
                        tot_solves += 1
                        resolve_times += 1
                        start_mp = time.time()
                        lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_resolve(env, self.K, tau, model, TL = mTL)
                        mp_time += time.time() - start_mp
                        if status == 'time_limit':
                            TL_reached += 1
                            fx  = None 
                            fy  = None
                            fys = None 
                        elif status == 'optimal' : 
                            fx  = copy.deepcopy(x)
                            fy  = copy.deepcopy(y)
                            fys = copy.deepcopy(ys)
                            true_lb = lb
                            prune_count, prune_state = self.check_prune_by_bound(true_lb, UB_i, prune_count, tot_nodes)
                            if prune_state :
                                new_model = True
                                k_new = None
                                model.dispose()
                                continue
                        # cannot be "status == 'infeasible", since we already have a feasible solution
            else :
                if new_model :
                    new_model_times += 1
                    tot_solves += 1
                    start_mp = time.time()
                    lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_build(env, gp_env, self.K, tau, k_new, fx, fy, fys, TL = mTL)
                    mp_time += time.time() - start_mp
                else :
                    tot_solves += 1
                    start_mp = time.time()                    
                    lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_update(env, model, self.K, tau, k_new, s, fx, fy, fys, TL = mTL)
                    mp_time += time.time() - start_mp
                
                if status == 'time_limit':
                        TL_reached += 1
                        fx  = None 
                        fy  = None
                        fys = None
                elif status == 'infeasible':
                    prune_count, prune_infeas_count, new_model = self.prune_by_infeas(prune_count, prune_infeas_count, tot_nodes)
                    k_new = None
                    model.dispose()
                    continue 
                elif status == 'optimal' :
                    if fixx : 
                        fx  = copy.deepcopy(x)
                        fy  = copy.deepcopy(y)
                        fys =  copy.deepcopy(ys)
                    true_lb = lb
                    prune_count, prune_state = self.check_prune_by_bound(true_lb, UB_i, prune_count, tot_nodes)
                    if prune_state :
                        new_model = True
                        k_new = None
                        model.dispose()
                        continue

            k_new = None

            if not status == 'time_limit' :
                start_sp = time.time()
                feas, partition, ub_tau, ngc_count, model, ngc_list, s_feas = self.induced_partition(env, model, tau, x, y, obj_R, unseen_scen, M, ngc_count, lazy, ngc_list)
                sp_time += time.time() - start_sp
            
                if feas :
                    status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(ub_tau, x, y, partition, start_time, 
                                                                                                    UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                    if status_update :
                        prune_count, prune_state = self.check_prune_by_bound(lb, UB_i, prune_count, tot_nodes)
                        if prune_state : 
                            if true_lb is not None : 
                                new_model = True
                                model.dispose()
                                continue 
                            else : 
                                resolve_times += 1
                                tot_solves += 1
                                prune_count -= 1
                                start_mp = time.time()
                                lb, x, y, ys, model, obj_R, s_new, status = self.problem.spip_resolve(env, self.K, tau, model, TL = mTL)
                                mp_time += time.time() - start_mp
                                if status == 'time_limit':
                                    TL_reached += 1
                                    fx  = None 
                                    fy  = None
                                    fys = None
                                else :
                                    true_lb = lb

                                    start_sp = time.time()
                                    feas, partition, ub_tau, ngc_count, model, ngc_list, s_feas = self.induced_partition(env, model, tau, x, y, obj_R, unseen_scen, M, ngc_count, lazy, ngc_list)
                                    sp_time += time.time() - start_sp

                                    if feas :
                                        status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(ub_tau, x, y, partition, start_time, 
                                                                                                        UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                                        if status_update :
                                            prune_count, prune_state = self.check_prune_by_bound(true_lb, UB_i, prune_count, tot_nodes)
                                            if prune_state : 
                                                new_model = True
                                                model.dispose()
                                                continue

            # we prepare to build the child nodes :
            j += 1
            u -= 1

            # scenario selection strategy :
            s = self.scen_selection(rng, status, s_sel, unseen_scen, s_new, s_feas)
            
            if u == 0 :
                leaf = True
                fx  = None
                fy  = None
                fys = None
                K_prime = min(self.K, full_list[-1] + 2)
                for k_new in range(K_prime) :
                    tot_nodes  += 1
                    tot_leaves += 1
                    tot_solves += 1
                    tau[k_new].append(s)
                    TL = self.max_time
                    start_mp = time.time()
                    lb, x, y, _,  _, _, _, status = self.problem.spip_update(env, model, self.K, tau, k_new, s, fx, fy, fys, leaf = leaf, TL = TL)
                    mp_time += time.time() - start_mp
                    if status == 'time_limit':
                        TL_reached += 1
                        print('time limit reached')
                        return tau_i, {"time": time.time() - start_time, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "tot_solves": tot_solves, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram                       
                    elif status == 'optimal' :
                        status_update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y = self.check_update_inc(lb, x, y, tau, start_time, 
                                                                                                 UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y)
                        prune_count += 1 
                    else :
                        prune_count, prune_infeas_count, new_model = self.prune_by_infeas(prune_count, prune_infeas_count, tot_nodes)

                    tau[k_new].remove(s)

                k_new = None
                new_model = True
                leaf = False
                model.dispose()
                continue
            else :
                N_set, k_new, s, skip = self.child_generation(M, s, rng, N_set, tau, fixx, fx, fy, fys, true_lb, j, y, status)
                new_model = False

        runtime = time.time() - start_time
        inc_UB_t[runtime] = UB_i

        print(f"spip_update had to be rerun {resolve_times} times over {solved_fixing_vars}")
        print(f"recording the lb avoided the solution of {lb_useful} opt models")
        print(f"the time limit at a node was reached {TL_reached} times over {tot_nodes}")
        print(f"we generated {ngc_count} no good cuts")
        if self.print_info:
            now = datetime.now().time()
            now_nice = f"{now.hour}:{now.minute}:{now.second}"
            print(f"Instance R {env.inst_num}, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")
            print(f"Total number of nodes visited : {tot_nodes}")
            print(f"Total prunings : {prune_count},  prunings by infeasibility : {prune_infeas_count}")
            print(f"The optimal partition is : {tau_i}")
            print(f"The opt first stage var is: {inc_x[next(reversed(inc_x))]}")
            print(f"The opt second stage policies are {inc_y[next(reversed(inc_y))]}")

        return tau_i, {"time": runtime, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": prune_infeas_count, "tot_leaves": tot_leaves, "tot_solves": tot_solves, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram

    ########################################################################################################

    def check_update_inc(self, new_ub, new_x, new_y, new_tau, start_time, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y) :
        update = False
        if new_ub < UB_i :
            UB_i, x_i, y_i = ( new_ub, copy.deepcopy(new_x), copy.deepcopy(new_y))
            tau_i = {k : new_tau[k] for k in range(self.K)}
            inc_UB_t[time.time() - start_time] = UB_i
            inc_x[time.time() - start_time] = x_i
            inc_y[time.time() - start_time] = y_i
            update = True
        return update, UB_i, x_i, y_i, tau_i, inc_UB_t, inc_x, inc_y 
    
    def check_prune_by_bound(self, new_lb, UB, prune_count, tot_nodes) : 
        new_model = False
        if new_lb >= UB + self.tol :
            prune_count += 1 
            new_model = True
            if self.print_info :
                print(f'Prune by bound at node {tot_nodes}')
        return prune_count, new_model
        
    def prune_by_infeas(self, prune_count, prune_infeas_count, tot_nodes) : 
        prune_count += 1
        prune_infeas_count += 1
        new_model = True
        if self.print_info :
            print(f'Prune by infeasibility at node {tot_nodes}')
        return prune_count, prune_infeas_count, new_model
    
    def scen_selection(self, rng, status, s_sel, unseen_scen, s_new=None, s_feas=None):
        if status == 'time_limit':
            s = int(rng.choice(unseen_scen))
        elif s_sel == 'sfeas' and s_feas is not None:
            s = s_feas
        elif s_new is not None:
            s = s_new
        else: 
            s = int(rng.choice(unseen_scen))
        return s
    
    def child_generation(self, M, s_new, rng, N_set, old_tau, fixx, fx, fy, fys, old_lb, depth, y, status) :
        skip = False
        K_prime = min(self.K, M + 1)
        K_set = np.arange(K_prime)

        if M == self.K: 
            k_new = rng.randint(self.K) 
        else:
            k_new = K_set[-1] 
            if not(status == 'time_limit') and (old_lb is not None) :
                skip = True

        for k in K_set:
            if k == k_new :
                continue
            tau_all_tmp = {}
            tau_all_tmp['tau'] = copy.deepcopy(old_tau)
            tau_all_tmp['snew'] = s_new
            tau_all_tmp['knew'] = k
            if fixx :
                tau_all_tmp['fx']  = copy.deepcopy(fx)
                tau_all_tmp['fy']  = copy.deepcopy(fy)
                tau_all_tmp['fys'] = copy.deepcopy(fys)
            tau_all_tmp['lb'] = old_lb
            tau_all_tmp['j'] = depth 

            N_set.append(tau_all_tmp)
        return N_set, k_new, s_new, skip
    
    def induced_partition(self, env, model, tau, x, y, obj_R, unseen_scen, M, ngc_count, lazy, ngc_list) :
        partition = {k: [] for k in range(self.K)}
        ub_tau = obj_R
        ind_k = {}
        s_feas = None
        for ss in range(env.N_scen) :
            feas, ind_k[ss], ob_ss = self.problem.sip(env, M, x, y, ss)
            if feas :
                partition[ind_k[ss]].append(ss)
                ub_tau += ob_ss
            else :
                if ss in unseen_scen :
                    s_feas = ss
                if M == self.K and lazy:
                    lhs_expr, rhs_const = self.problem.build_no_good_cut(env, model, self.K, x, y)
                    ngc_count += 1
                    model._lazy_constraints.append((lhs_expr, rhs_const)) 
                    ngc_list.append((lhs_expr, rhs_const))
                break
        return feas, partition, ub_tau, ngc_count, model, ngc_list, s_feas

    def solveMiqp(self) :
        env = copy.deepcopy(self.env) 
        gp_env = gp.Env()

        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("Threads", 1)

        now = datetime.now().time()
        print(f"Instance {env.inst_num}: miqp started at {now}")
        start_time = time.time()
        feas, fopt, x, y, clusters, inc_UB_t = self.problem.miqp(env, gp_env,self.K)
        runtime = time.time() - start_time
        inc_UB_t[runtime] = fopt

        if not feas :
            print("compact not feasible")
            return None, {"time": runtime, "f_kopt": np.inf, "tot_nodes": 0, "n_prunes": 0, "prune_infeas_count": 0, "tot_leaves":0, "tot_solves": 0, "inc_ub_t": inc_UB_t}, psutil.virtual_memory().percent > self.ram

        if self.print_info:
            now = datetime.now().time()
            now_nice = f"{now.hour}:{now.minute}:{now.second}"
            print(f"Instance R {env.inst_num}, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")

            print(f"The optimal partition is : {clusters}")
            print(f"The optimal second stage strategies are :")
            print(y)

        return clusters, {"time": runtime, "f_kopt": fopt, "tot_nodes": 0, "n_prunes": 0, "prune_infeas_count": 0, "tot_leaves":0, "tot_solves": 0, 'inc_ub_t': inc_UB_t}, psutil.virtual_memory().percent > self.ram

    def solveCompleteEnum(self) :
        env = copy.deepcopy(self.env)
        gp_env = gp.Env()

        gp_env.setParam("OutputFlag", 0)
        gp_env.setParam("Threads", 1)

        prune_count = 0
        tot_nodes  = 1
        tot_leaves = 0
        mp_time = 0

        UB_i = env.upper_bound
        lb   = env.lower_bound

        inc_UB_t = {}

        now = datetime.now().time()
        print(f"Instance {env.inst_num}: complete_enum started at {now}")
        start_time = time.time()
        tau = {k: [] for k in range(self.K)}
        s_new = 0
        tau[0].append(s_new) # we define tau = {0:[s_new], 1:[],..., K:[]}
        N_set = [tau]

        while not(N_set == []) and time.time() - start_time < self.max_time:
            tot_nodes += 1
            tau = N_set.pop()
            seen   = [s for k in range(self.K) for s in tau[k]] # list of scenarios in tau
            unseen = [s for s in range(env.N_scen) if s not in seen] # list of scenarios not in tau
            u = len(unseen)
            full_list = [k for k in range(self.K) if len(tau[k]) > 0]
            M = len(full_list) # number of non-empty sets / 2nd stage strategies actually computed

            if M < self.K and u == self.K - M:
                #tot_nodes += 1
                for k in range(self.K-M):
                    tau[M+k].append(unseen[k])
                start_mp = time.time()
                lb, _, _, _, _,_, _,status = self.problem.spip_build(env, gp_env, self.K, tau, TL = self.max_time)
                mp_time += time.time() - start_mp
                if status == 'optimal' and lb < UB_i:
                    tau_i = copy.deepcopy(tau)
                    UB_i = copy.deepcopy(lb)
                    inc_UB_t[time.time() - start_time] = UB_i
                prune_count += 1
                tot_leaves += 1
                continue
            elif u == 0 :
                start_mp = time.time()
                lb, _, _, _, _,_, _,status = self.problem.spip_build(env, gp_env, self.K, tau, TL = self.max_time)
                mp_time += time.time() - start_mp
                if status == 'optimal' and lb < UB_i :
                    tau_i = copy.deepcopy(tau)
                    UB_i  = copy.deepcopy(lb)
                    inc_UB_t[time.time() - start_time] = UB_i
                prune_count += 1
                tot_leaves  += 1
                continue
            else :
                s_new = unseen.pop()

            if len(full_list) == 0: 
                K_set = [0]
            elif len(full_list) == self.K:
                K_set = np.arange(self.K)   
            else: 
                K_prime = min(self.K, full_list[-1] + 2) 
                K_set = np.arange(K_prime)

            for k in K_set:
                tau_tmp = copy.deepcopy(tau)
                tau_tmp[k].append(s_new)
                N_set.append(tau_tmp)

        runtime = time.time() - start_time
        inc_UB_t[runtime] = UB_i

        if self.print_info:
            now = datetime.now().time()
            now_nice = f"{now.hour}:{now.minute}:{now.second}"
            print(f"Instance R {env.inst_num}, completed at {now_nice}, solved in {np.round(runtime/60, 3)} minutes")
            print(f"Total number of nodes visited : {tot_nodes}")

        return tau_i, {"time": runtime, "f_kopt": UB_i, "tot_nodes": tot_nodes, "n_prunes": prune_count, "prune_infeas_count": 0, "tot_leaves": tot_leaves, "tot_solves": tot_nodes, 'inc_ub_t': inc_UB_t}, psutil.virtual_memory().percent > self.ram

    def not_in_tau(self, env, tau) :
        seen   = [s for k in range(self.K) for s in tau[k]] # list of scenarios in tau
        unseen = [s for s in range(env.N_scen) if s not in seen] # list of scenarios not in tau

        return unseen, len(unseen)