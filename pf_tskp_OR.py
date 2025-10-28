import numpy as np
import gurobipy as gp
from gurobipy import GRB
import problem_functions
import copy

class tskp_Inst(problem_functions.Pb_functions):
    def __init__(self, tmax):
        self.tmax = tmax 

    def complete(self, env, gp_env) :
        model = gp.Model("Complete two-stage stochastic model", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        # variables
        x = model.addVars(env.r, vtype = GRB.BINARY, name = "x")
        y = {}
        for s in range(env.N_scen):
            y[s] = model.addVars(env.N, vtype=GRB.BINARY, name=f"y[{s}]")

        # objective function
        obj_expr = - gp.quicksum(env.c[h]*x[h] for h in range(env.r)) + gp.quicksum(gp.quicksum(env.cost[s][i] * y[s][i] for i in range(env.N))*env.prob[s] for s in range(env.N_scen))
        model.setObjective(- obj_expr, GRB.MINIMIZE)

        # constraints
        model.addConstrs(gp.quicksum(env.d[h]*x[h] for h in range(env.r)) + gp.quicksum(y[s][i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s] for s in range(env.N_scen))
        for h in range(env.r):
            model.addConstrs(y[s][h*env.q + i]<=x[h] for i in range(env.q) for s in range(env.N_scen))

        # solve
        model.update()
        model.optimize()

        if not( model.Status == gp.GRB.OPTIMAL ):
            if not (model.Status == gp.GRB.INFEASIBLE) :
                print(f"Model status is {model.Status}")
            else : 
                status = 'infeasible'
            if model.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
            obj_val = None
            x_sol = None
            y_sol = None
        else :
            status = 'optimal'
            obj_val = model.ObjVal
            x_sol = {i: var.X for i, var in x.items()}
            y_sol = {}
            for s in range(env.N_scen) : 
                y_sol[s] = {i: var.X for i, var in y[s].items()}
                
        return obj_val, x_sol, y_sol, status 

    def spip_build(self, env, gp_env, K, tau, k_new = None, x_val = None, y_val = None, y_s_val = None, TL = None, ngc_list_tmp = None):
        spip = gp.Model("Scenario-Based K-Adaptability Problem", env = gp_env)
        # model parameters:
        spip.Params.OutputFlag = 0
        if TL is None : 
            spip.Params.TimeLimit = self.tmax
        else :
            spip.Params.TimeLimit = TL
        spip.Params.LazyConstraints = 1

        if ngc_list_tmp is None : 
            ngc_list_tmp = []

        spip._lazy_constraints = ngc_list_tmp
 
        list_scen   = [s for k in range(K) for s in tau[k]]
        unseen_scen = [s for s in range(env.N_scen) if s not in list_scen ]

        # variables
        x = spip.addVars(env.r, vtype = GRB.BINARY, name = "x")
        y = {}
        for k in range(K):
            y[k] = spip.addVars(env.N, vtype=GRB.BINARY, name=f"y[{k}]")
        y_s = {}
        for s in unseen_scen : 
            y_s[s] = spip.addVars(env.N, vtype=GRB.BINARY, name=f"y_s[{s}]" )

        if x_val is not None :
            for h in range(env.r) : 
                x[h].LB = x_val[h]
                x[h].UB = x_val[h]

        if y_val is not None :
            for k in range(K) :
                if tau[k] : 
                    if not k == k_new : 
                        for i in range(env.N) : 
                            y[k][i].LB = y_val[k][i]
                            y[k][i].UB = y_val[k][i]

        if y_s_val is not None :
            for s in unseen_scen:
                for i in range(env.N) : 
                    y_s[s][i].LB = y_s_val[s][i]
                    y_s[s][i].UB = y_s_val[s][i]
        
        # objective function
        obj_expr = - gp.quicksum(env.c[h]*x[h] for h in range(env.r)) + \
                        gp.quicksum(gp.quicksum(gp.quicksum(env.cost[s][i] * y[k][i] for i in range(env.N)) * env.prob[s]
                                                for s in tau[k]) for k in range(K) if tau[k]) + \
                        gp.quicksum(gp.quicksum(env.cost[s][i] * y_s[s][i] for i in range(env.N)) * env.prob[s] for s in unseen_scen)
        spip.setObjective( -obj_expr, GRB.MINIMIZE)

        # constraints
        con_capacity = {}
        for k in range(K):
            for h in range(env.r):
                    spip.addConstrs(y[k][h*env.q + i]<=x[h] for i in range(env.q))
            if len(tau[k]) :
                for s in tau[k] : 
                    con_capacity[k,s] = spip.addConstr(gp.quicksum(env.d[h]*x[h] for h in range(env.r)) +
                                                       gp.quicksum(y[k][i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s], name = f"con_capacity[{k},{s}]" )
        con_capacity_s = {}
        con_x_y_s = {}
        for s in unseen_scen :
            con_capacity_s[s] = spip.addConstr(gp.quicksum(env.d[h]*x[h] for h in range(env.r)) +
                                                       gp.quicksum(y_s[s][i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s], name = f"con_capacity_s[{s}]" )
            for h in range(env.r):
                for i in range(env.q) :
                    con_x_y_s[s,h,i] = spip.addConstr(y_s[s][h*env.q + i]<=x[h], name = f"con_x_y_s[{s},{h},{i}]" )

        # solve
        spip.update()
        if spip._lazy_constraints == [] :
            spip.optimize()
        else :
            spip.optimize(self.lazy_callback)

        if not(spip.Status == gp.GRB.OPTIMAL):
            if spip.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
                obj_val = None
            elif spip.Status == gp.GRB.INFEASIBLE : 
                status = 'infeasible'
                obj_val = np.inf
            else :
                print(f'Error : Model status is {spip.Status}')
            x_sol = None
            y_sol = None
            y_s_sol   = None
            obj_val_R = None
            s_max = None
        else :
            status = 'optimal'
            obj_val = spip.ObjVal  
            x_sol = {i: var.X for i, var in x.items()}
            y_sol = {}
            y_s_sol  = {}
            max_ob_s = - np.inf
            s_max = env.N_scen
            obj_s = None
            obj_val_R = sum(env.c[h]*x_sol[h] for h in range(env.r))
            for k in range(K):
                y_sol[k] = {i: var.X for i, var in y[k].items()}
            for s in unseen_scen : 
                y_s_sol[s] = {i: var.X for i, var in y_s[s].items()}
                obj_s = sum(env.cost[s][i] * y_s_sol[s][i] for i in range(env.N)) * env.prob[s]
                if obj_s > max_ob_s : 
                    max_ob_s = obj_s
                    s_max = s
      
        re_update = False

        if x_val is not None :
            re_update = True
            for h in range(env.r) : 
                x[h].LB = 0.0
                x[h].UB = 1.0

        if y_val is not None :
            re_update = True
            for k in range(K) :
                for i in range(env.N) : 
                    y[k][i].LB = 0.0
                    y[k][i].UB = 1.0

        if y_s_val is not None :
            re_update = True
            for s in unseen_scen:
                for i in range(env.N) : 
                    y_s[s][i].LB = 0.0
                    y_s[s][i].UB = 1.0
        
        if re_update :
            spip.update()
            
        return obj_val, x_sol, y_sol, y_s_sol, spip, obj_val_R, s_max, status
    
    def spip_update(self, env, spip, K, tau, k_new, s_new, x_val, y_val, y_s_val, leaf = False, skip = False, TL = None):
        # load variables
        x = {h: spip.getVarByName(f"x[{h}]") for h in range(env.r)}
        y = {k: {i: spip.getVarByName(f"y[{k}][{i}]") for i in range(env.N)} for k in range(K)}

        list_scen   = [s for k in range(K) for s in tau[k]]
        unseen_scen = [s for s in range(env.N_scen) if s not in list_scen]

        y_s = {s: {i: spip.getVarByName(f"y_s[{s}][{i}]") for i in range(env.N)} for s in unseen_scen}

        assert k_new is not None

        if not skip :
            if x_val is not None :
                for h in range(env.r) : 
                    x[h].LB = x_val[h]
                    x[h].UB = x_val[h]

            if y_val is not None :
                for k in range(K) :
                    if tau[k] :
                        if not k == k_new : 
                            for i in range(env.N) : 
                                y[k][i].LB = y_val[k][i]
                                y[k][i].UB = y_val[k][i]

            if y_s_val is not None :
                for s in unseen_scen :
                    for i in range(env.N) : 
                        y_s[s][i].LB = y_s_val[s][i]
                        y_s[s][i].UB = y_s_val[s][i]

        for i in range(env.N) :
            spip.remove(spip.getVarByName(f"y_s[{s_new}][{i}]")) 
            y[k_new][i].Obj = y[k_new][i].Obj - env.cost[s_new][i] * env.prob[s_new] 

        # add constraints :
        con_capacity = {}
        con_capacity[k_new,s_new] = spip.addConstr(gp.quicksum(env.d[h]*x[h] for h in range(env.r)) + gp.quicksum(y[k_new][i]*env.weight[s_new][i] for i in range(env.N)) <= env.capacity[s_new], name = f"con_capacity[{k_new},{s_new}]")

        temp_con_to_remove_s = []
        temp_con_to_remove_s.append(spip.getConstrByName(f"con_capacity_s[{s_new}]"))
        for h in range(env.r) :
            for i in range(env.q) :
                temp_con_to_remove_s.append(spip.getConstrByName(f"con_x_y_s[{s_new},{h},{i}]"))

        if temp_con_to_remove_s :
            spip.remove(temp_con_to_remove_s)
            
        # update
        if TL is None : 
            spip.Params.TimeLimit = self.tmax
        else :
            spip.Params.TimeLimit = TL
        spip.update()

        if skip :
            assert y_s_val is not None
            y_val[k_new] = copy.deepcopy(y_s_val[s_new]) # updated y 
            del y_s_val[s_new] # updated ys
            status = 'optimal'
            obj_val_R = sum(env.c[h]*x_val[h] for h in range(env.r)) 
            obj_val = obj_val_R - sum(sum(sum(env.cost[s][i] * y_val[k][i] for i in range(env.N)) * env.prob[s]
                                                    for s in tau[k]) for k in range(K) if tau[k])
            # scenario selection
            max_ob_s = - np.inf
            s_max = env.N_scen
            for s in unseen_scen :
                obj_s = sum(env.cost[s][i] * y_s_val[s][i] for i in range(env.N)) * env.prob[s]
                obj_val -= obj_s
                if obj_s > max_ob_s :
                    max_ob_s = obj_s
                    s_max = s

            return obj_val, x_val, y_val, y_s_val, spip, obj_val_R, s_max, status
        else : 
            if spip._lazy_constraints == [] :
                spip.optimize()
            else :
                spip.optimize(self.lazy_callback)

            if not(spip.Status == gp.GRB.OPTIMAL):
                if spip.Status == gp.GRB.TIME_LIMIT : 
                    status = 'time_limit'
                    obj_val = None
                elif spip.Status == gp.GRB.INFEASIBLE : 
                    status = 'infeasible'
                    obj_val = np.inf
                else :
                    print(f'Error : Model status is {spip.Status}')
                x_sol = None
                y_sol = None
                y_s_sol = None
                obj_val_R = None
                s_max = None
            else :
                status = 'optimal'
                obj_val = spip.ObjVal  
                x_sol = {i: var.X for i, var in x.items()}
                y_sol = {}
                y_s_sol = {}
                max_ob_s = - np.inf
                s_max = env.N_scen
                obj_val_R = sum(env.c[h]*x_sol[h] for h in range(env.r))
                for k in range(K):
                    y_sol[k] = {i: var.X for i, var in y[k].items()}
                for s in unseen_scen : 
                    y_s_sol[s] = {i: var.X for i, var in y_s[s].items()}
                    obj_s = sum(env.cost[s][i] * y_s_sol[s][i] for i in range(env.N)) * env.prob[s]
                    if obj_s > max_ob_s : 
                        max_ob_s = obj_s
                        s_max = s

            re_update = False

            if x_val is not None :
                re_update = True
                for h in range(env.r) : 
                    x[h].LB = 0.0
                    x[h].UB = 1.0

            if y_val is not None :
                re_update = True
                for k in range(K) :
                    for i in range(env.N) : 
                        y[k][i].LB = 0.0
                        y[k][i].UB = 1.0

            if y_s_val is not None :
                re_update = True
                for s in unseen_scen:
                    for i in range(env.N) : 
                        y_s[s][i].LB = 0.0
                        y_s[s][i].UB = 1.0

            if leaf :
                if k_new is not None :
                    for i in range(env.N):
                        y[k_new][i].Obj = y[k_new][i].Obj + env.cost[s_new][i] * env.prob[s_new]
                    y_s = {}
                    y_s[s_new] = spip.addVars(env.N, vtype=GRB.BINARY, name=f"y_s[{s_new}]" )
                    spip.addConstr(gp.quicksum(env.d[h]*x[h] for h in range(env.r)) +
                                                        gp.quicksum(y_s[s_new][i]*env.weight[s_new][i] for i in range(env.N)) <= env.capacity[s_new], name = f"con_capacity_s[{s_new}]" )
                    for h in range(env.r):
                        for i in range(env.q) :
                            spip.addConstr(y_s[s_new][h*env.q + i]<=x[h], name = f"con_x_y_s[{s_new},{h},{i}]" )

                    # remove constraints :
                    spip.remove(con_capacity[k_new,s_new])

                    re_update = True 
                s_max = s_new
            
            if re_update :
                spip.update()
                
            return obj_val, x_sol, y_sol, y_s_sol, spip, obj_val_R, s_max, status 

    def spip_resolve(self, env, K, tau, spip, TL = None):
        x = {h: spip.getVarByName(f"x[{h}]") for h in range(env.r)}
        y = {k: {i: spip.getVarByName(f"y[{k}][{i}]") for i in range(env.N)} for k in range(K)}

        list_scen = [s for k in range(K) for s in tau[k]]
        unseen_scen = [s for s in range(env.N_scen) if s not in list_scen ]

        y_s = {s: {i: spip.getVarByName(f"y_s[{s}][{i}]") for i in range(env.N)} for s in unseen_scen}

        if TL is None : 
            spip.Params.TimeLimit = self.tmax
        else :
            spip.Params.TimeLimit = TL

        if spip._lazy_constraints == [] :
            spip.optimize()
        else :
            spip.optimize(self.lazy_callback)

        if not(spip.Status == gp.GRB.OPTIMAL):
            if spip.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
                obj_val = None
            elif spip.Status == gp.GRB.INFEASIBLE : 
                status = 'infeasible'
                obj_val = np.inf
            else :
                print(f'Error : Model status is {spip.Status}')
            x_sol = None
            y_sol = None
            y_s_sol = None
            obj_val_R = None
            s_max = None
        else :
            status = 'optimal'
            obj_val = spip.ObjVal
            x_sol = {h: spip.getVarByName(f"x[{h}]").X for h in range(env.r)}
            y_sol = {k: {i: spip.getVarByName(f"y[{k}][{i}]").X for i in range(env.N)} for k in range(K)}
            y_s_sol = {s : {i: spip.getVarByName(f"y_s[{s}][{i}]").X for i in range(env.N)} for s in unseen_scen}
            max_ob_s = - np.inf
            s_max = env.N_scen
            for s in unseen_scen:
                obj_s = sum(env.cost[s][i] * y_s_sol[s][i] for i in range(env.N)) * env.prob[s]
                if obj_s > max_ob_s : 
                    max_ob_s = obj_s
                    s_max = s
            obj_val_R = sum(env.c[h]*x_sol[h] for h in range(env.r)) 
            
        return obj_val, x_sol, y_sol, y_s_sol, spip, obj_val_R, s_max, status

    def lazy_callback(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            if model._lazy_constraints : 
                for lhs_expr, rhs_constant in model._lazy_constraints:
                    lhs_val = 0.0
                    for i in range(lhs_expr.size()):
                        var = lhs_expr.getVar(i)     
                        coeff = lhs_expr.getCoeff(i)
                        lhs_val += coeff * model.cbGetSolution(var) 

                    if lhs_val > rhs_constant + 1e-6:
                        model.cbLazy(lhs_expr <= rhs_constant)
        else :
            return
                
    
    def build_no_good_cut(self, env, model, K, x_sol, y_sol):
        x = {h: model.getVarByName(f"x[{h}]") for h in range(env.r)}
        y = {k: {i: model.getVarByName(f"y[{k}][{i}]") for i in range(env.N)} for k in range(K)}

        Sx = [h for h in range(env.r) if x_sol[h] >= 1 - 1e-6]
        Tx = [h for h in range(env.r) if x_sol[h] <= 1e-6]

        LSx = len(Sx)

        Sy = {k: [] for k in range(K)}
        Ty = {k: [] for k in range(K)}

        LSy = 0

        for k in range(K):
            Sy[k] = [i for i in range(env.N) if y_sol[k][i] >= 1 - 1e-6]
            Ty[k] = [i for i in range(env.N) if y_sol[k][i] <= 1e-6]
            LSy += len(Sy[k])

        lhs = gp.LinExpr()

        for h in range(env.r): 
            if h in Sx:
                lhs += x[h]
            else:
                lhs -= x[h]

        for k in range(K):
            for i in range(env.N):
                if i in Sy[k]:
                    lhs += y[k][i]
                else :
                    lhs -= y[k][i]

        rhs = LSx + LSy - 1

        return lhs, rhs

    def sip(self, env, K, x, y, s):
        Klist = [k for k in range(K) 
                    if sum(env.d[h]*x[h] for h in range(env.r)) + sum(y[k][i] * env.weight[s][i] for i in range(env.N)) <= env.capacity[s]]

        if Klist == [] : 
            return False, None, None
        else :
            best_ob = np.inf
            j = K 
            for k in Klist :
                obk = - sum(env.cost[s][i] *y[k][i] for i in range(env.N)) * env.prob[s]
                if obk < best_ob : 
                    best_ob = obk
                    j = k

            return True, j, best_ob
        
    def miqp(self, env, gp_env, K) :
        compact = gp.Model("K-adaptability as a quadratic program", env = gp_env)
        # model parameters
        compact.Params.OutputFlag = 0
        compact.setParam("TimeLimit", self.tmax)
        compact.Params.Threads = 1

        x = compact.addVars(env.r, vtype = GRB.BINARY, name = "x")
        y = {}
        u = {}
        for j in range(K):
            y[j] = compact.addVars(env.N, vtype=GRB.BINARY, name=f"y[{j}]")
            for s in range(env.N_scen):
                u[s,j] = compact.addVar(vtype=GRB.BINARY, name = f"u[{s},{j}]")

        # objective function
        obj_expr = - gp.quicksum(env.c[h]*x[h] for h in range(env.r)) +  gp.quicksum( gp.quicksum(env.cost[s][i] * gp.quicksum(y[j][i]*u[s,j] for j in range(K)) for i in range(env.N))*env.prob[s] for s in range(env.N_scen))
        compact.setObjective( -obj_expr, GRB.MINIMIZE)

        # constraints
        for s in range(env.N_scen):
            for j in range(K):
                compact.addGenConstrIndicator(u[s,j], True, gp.quicksum(env.d[h]*x[h] for h in range(env.r)) +
                                       gp.quicksum(y[j][i]*env.weight[s][i] for i in range(env.N)) <= env.capacity[s] )
        for j in range(K) :
            for h in range(env.r):
                for i in range(env.q) :
                    compact.addConstr(y[j][h*env.q + i] <= x[h])

        compact.addConstrs(gp.quicksum(u[s,j] for j in range(K)) == 1 for s in range(env.N_scen))

        # solve
        self.inc_UB = {}
        compact.update()
        compact.optimize(self.my_callback)

        if not(compact.Status == gp.GRB.OPTIMAL):
            if not (compact.Status == gp.GRB.INFEASIBLE) :
                print(f"Model status is {compact.Status}")
            feas    = False
            obj_val = None
            x_sol   = None
            y_sol   = None
            u_sol   = None
        else :
            feas = True
            obj_val = compact.ObjVal  
            x_sol = {i: var.X for i, var in x.items()}
            y_sol = {}
            u_sol = {}
            for j in range(K) : 
                y_sol[j] = {i: var.X for i, var in y[j].items()}
            for j in range(K):
                u_sol[j] = [s  for s in range(env.N_scen) if u[s,j].X==1]
            
        return feas, obj_val, x_sol, y_sol, u_sol, self.inc_UB

    def my_callback(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # Time (in seconds since start)
            time = model.cbGet(gp.GRB.Callback.RUNTIME)
            # Best incumbent objective (upper bound)
            ub = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            
            # Save if not already saved at that time
            self.inc_UB[time] = ub
