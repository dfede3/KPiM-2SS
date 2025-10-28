import numpy as np
import gurobipy as gp
from gurobipy import GRB
import problem_functions
import copy

class kp_Inst(problem_functions.Pb_functions):
    def __init__(self, tmax):
        self.tmax = tmax 

    def complete(self, env, gp_env, TL = None) :
        
        model = gp.Model("Complete stochastic model", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        if TL is None : 
            model.Params.TimeLimit = self.tmax
        else :
            model.Params.TimeLimit   = TL

        # variables 
        y = {} 
        for s in range(env.N_scen):
            y[s] = model.addVars(env.N, vtype=GRB.BINARY, name=f"y[{s}]")

        # objective function 
        obj_expr = gp.quicksum(gp.quicksum(env.cost[s][i] * y[s][i] for i in range(env.N))*env.prob[s] for s in range(env.N_scen))
        model.setObjective(- obj_expr, GRB.MINIMIZE)

        # constraints
        model.addConstrs(gp.quicksum(y[s][i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s] for s in range(env.N_scen))

        # solve
        model.update()
        model.optimize()

        if not(model.Status == gp.GRB.OPTIMAL):
            if not (model.Status == gp.GRB.INFEASIBLE) :
                print(f"Model status is {model.Status}")
            else : 
                status = 'infeasible'
            if model.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
            obj_val = None
            y_sol = None
            scen_obj = None
            s_max = None
        else :
            status = 'optimal'
            obj_val = model.ObjVal
            y_sol = {}
            scen_obj = {}
            s_max = env.N_scen
            max_ob_s = - np.inf 
            for s in range(env.N_scen) : 
                y_sol[s] = {i: var.X for i, var in y[s].items()}
                scen_obj[s] = - sum(env.cost[s][i] * y_sol[s][i] for i in range(env.N))*env.prob[s]
                if scen_obj[s] > max_ob_s : 
                    max_ob_s = scen_obj[s]
                    s_max = s

        return obj_val, y_sol, status, scen_obj, s_max

    def pip(self, env, gp_env, K, tau, TL = None):
        spip = gp.Model("Scenario-Based K-Adaptability Problem", env = gp_env)
        
        # model parameters:
        spip.Params.OutputFlag = 0
        if TL is None : 
            spip.Params.TimeLimit = self.tmax
        else :
            spip.Params.TimeLimit = TL
    
        list_scen   = [s for k in range(K) for s in tau[k]]
        unseen_scen = [s for s in range(env.N_scen) if s not in list_scen ]

        # variables 
        y = {} # K second stage variables 
        for k in range(K):
            y[k] = spip.addVars(env.N, vtype=GRB.BINARY, name=f"y[{k}]")

        # objective function
        obj_expr = gp.quicksum(gp.quicksum(env.cost[s][i] * env.prob[s] for s in tau[k]) * y[k][i] for k in range(K) if tau[k] for i in range(env.N))
        spip.setObjective( -obj_expr, GRB.MINIMIZE)

        # constraints
        con_capacity = {}
        for k in range(K):
            if len(tau[k]) :
                for s in tau[k] :
                    con_capacity[k,s] = spip.addConstr(gp.quicksum(y[k][i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s], name = f"con_capacity[{k},{s}]")

        # solve
        spip.update()
        spip.optimize()

        if not(spip.Status == gp.GRB.OPTIMAL):
            if spip.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
            elif spip.Status == gp.GRB.INFEASIBLE : 
                status = 'infeasible'
            else :
                print(f'Error : Model status is {spip.Status}')
            obj_val = None
            y_sol = None
        else :
            status = 'optimal'
            obj_val = spip.ObjVal  
            y_sol = {}
            for k in range(K):
                y_sol[k] = {i: var.X for i, var in y[k].items()}

        return obj_val, status

    def spip_build(self, env, gp_env, P, TL = None):
        spip = gp.Model("Scenario-Based K-Adaptability Problem", env = gp_env)
        
        # model parameters:
        spip.Params.OutputFlag = 0
        if TL is None : 
            spip.Params.TimeLimit = self.tmax
        else :
            spip.Params.TimeLimit = TL

        # variables 
        y = spip.addVars(env.N, vtype=GRB.BINARY, name=f"y")

        # objective function
        obj_expr = gp.quicksum(gp.quicksum(env.cost[s][i] * env.prob[s] for s in P) * y[i] for i in range(env.N))
        spip.setObjective( -obj_expr, GRB.MINIMIZE)

        # constraints
        con_capacity = {}
        for s in P :
            con_capacity[s] = spip.addConstr(gp.quicksum(y[i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s], name = f"con_capacity[{s}]")

        # solve
        spip.update()
        spip.optimize()

        if not(spip.Status == gp.GRB.OPTIMAL):
            if spip.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
            elif spip.Status == gp.GRB.INFEASIBLE : 
                status = 'infeasible'
            else :
                print(f'Error : Model status is {spip.Status}')
            obj_val = None
            y_sol = None
            s_max = None
        else :
            status = 'optimal'
            obj_val = spip.ObjVal  
            y_sol = {}
            y_sol = {i: var.X for i, var in y.items()}

        return obj_val, y_sol, spip, status
    
    def spip_update(self, env, spip, s_new, TL = None):
        # load variables
        y = {i: spip.getVarByName(f"y[{i}]") for i in range(env.N)}

        for i in range(env.N) :
            y[i].Obj = y[i].Obj - env.cost[s_new][i] * env.prob[s_new] # we update the value of the objective function for y[k_new][:]

        # add constraints :
        con_capacity = {}
        con_capacity[s_new] = spip.addConstr(gp.quicksum(y[i]*env.weight[s_new][i] for i in range(env.N)) <= env.capacity[s_new], name = f"con_capacity[{s_new}]") 

        # update
        if TL is None : 
            spip.Params.TimeLimit = self.tmax
        else :
            spip.Params.TimeLimit = TL
        spip.update()
        spip.optimize()


        if not(spip.Status == gp.GRB.OPTIMAL):
            if spip.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
                obj_val = None
            elif spip.Status == gp.GRB.INFEASIBLE : 
                status = 'infeasible'
                obj_val = np.inf
            else :
                print(f'Error : Model status is {spip.Status}')
            y_sol = None
        else :
            status = 'optimal'
            obj_val = spip.ObjVal  
            y_sol = {}
            y_sol = {i: var.X for i, var in y.items()}

        return obj_val, y_sol, spip, status

    def check_feas(self, env, y, s):
        if sum(y[i] * env.weight[s][i] for i in range(env.N)) <= env.capacity[s]:
            return True, - sum(env.cost[s][i] *y[i] for i in range(env.N)) * env.prob[s]
        else:
            return False, None

        
    def miqp(self, env, gp_env, K) :
        compact = gp.Model("K-adaptability as a quadratic program", env = gp_env)
        # model parameters
        compact.Params.OutputFlag = 0
        compact.setParam("TimeLimit", self.tmax)

        y = {}
        u = {}
        for j in range(K):
            y[j] = compact.addVars(env.N, vtype=GRB.BINARY, name=f"y[{j}]")
            for s in range(env.N_scen):
                u[s,j] = compact.addVar(vtype=GRB.BINARY, name = f"u[{s},{j}]")

        # objective function
        obj_expr = gp.quicksum( gp.quicksum(env.cost[s][i] * gp.quicksum(y[j][i]*u[s,j] for j in range(K)) for i in range(env.N))*env.prob[s] for s in range(env.N_scen))
        compact.setObjective( -obj_expr, GRB.MINIMIZE)

        # constraints
        for s in range(env.N_scen):
            for j in range(K):
                compact.addGenConstrIndicator(u[s,j], True, 
                                       gp.quicksum(y[j][i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s] )

        compact.addConstrs(gp.quicksum(u[s,j] for j in range(K)) == 1 for s in range(env.N_scen))

        # solve
        self.inc_UB = {}
        compact.update()
        compact.optimize()

        if not(compact.Status == gp.GRB.OPTIMAL):
            if not (compact.Status == gp.GRB.INFEASIBLE) :
                print(f"Model status is {compact.Status}")
            feas = False
            obj_val = None
            y_sol   = None
            u_sol   = None 
        else :
            feas = True
            obj_val = compact.ObjVal  
            y_sol = {}
            u_sol = {}
            for j in range(K) : 
                y_sol[j] = {i: var.X for i, var in y[j].items()}
            for j in range(K):
                u_sol[j] = [s  for s in range(env.N_scen) if u[s,j].X==1]
        return feas, obj_val, y_sol, u_sol, self.inc_UB
    
    def my_callback(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # Time (in seconds since start)
            time = model.cbGet(gp.GRB.Callback.RUNTIME)
            # Best incumbent objective (upper bound)
            ub = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            
            # Save if not already saved at that time
            self.inc_UB[time] = ub