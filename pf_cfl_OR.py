import numpy as np
import gurobipy as gp
from gurobipy import GRB
import problem_functions
import copy

class cfl_Inst(problem_functions.Pb_functions):
    def __init__(self, tmax):
        self.tmax = tmax 

    def complete(self, env, gp_env) :
        
        model = gp.Model("Complete two-stage stochastic cfl model", env = gp_env)
        # model parameters:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.tmax

        # variables
        x = model.addVars(env.r, vtype = GRB.BINARY, name = "x")
        y = {}
        for s in range(env.N_scen):
            y[s] = model.addVars(env.r, env.N, vtype=GRB.BINARY, name=f"y[{s}]")

        # objective function
        obj_expr = gp.quicksum(env.cost1[i]*x[i] for i in range(env.r)) + gp.quicksum(gp.quicksum(env.cost2[s,i][j] * y[s][i,j] for i in range(env.r) for j in range(env.N))*env.prob[s] for s in range(env.N_scen))
        model.setObjective(obj_expr, GRB.MINIMIZE)

        con_assignment = {}
        con_capacity = {}
        # constraints
        for s in range(env.N_scen) :
            for j in range(env.N):
                con_assignment[s,j] = model.addConstr(gp.quicksum(y[s][i,j] for i in range(env.r))==1, name = f'con_assignment[{s},{j}]')
            for i in range(env.r) :
                con_capacity[s,i] = model.addConstr(gp.quicksum(env.dem[s,i][j]*y[s][i,j] for j in range(env.N)) <= env.V[i]*x[i], name=f'con_capacity[{s},{i}]') 

        # solve
        model.update()
        model.optimize()

        if not( model.Status == gp.GRB.OPTIMAL ):
            if model.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
            elif model.Status == gp.GRB.INFEASIBLE : 
                status = 'infeasible'
            else :
                print(f'Error : Model status is {model.Status}')
            obj_val = None
            x_sol   = None
            y_sol   = None
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
            spip.Params.TimeLimit   = TL
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
            y[k] = spip.addVars(env.r, env.N, vtype=GRB.BINARY, name=f"y[{k}]")
        y_s = {}
        for s in unseen_scen : 
            y_s[s] = spip.addVars(env.r, env.N, vtype=GRB.BINARY, name=f"y_s[{s}]" )

        if x_val is not None :
            for i in range(env.r) : 
                x[i].LB = x_val[i]
                x[i].UB = x_val[i]

        if y_val is not None :
            for k in range(K) :
                if tau[k] : 
                    if not k == k_new : 
                        for i in range(env.r) :
                            for j in range(env.N) : 
                                y[k][i,j].LB = y_val[k][i,j]
                                y[k][i,j].UB = y_val[k][i,j]

        if y_s_val is not None :
            for s in unseen_scen:
                for i in range(env.r) :
                    for j in range(env.N) : 
                        y_s[s][i,j].LB = y_s_val[s][i,j]
                        y_s[s][i,j].UB = y_s_val[s][i,j]

        # objective function
        obj_expr = gp.quicksum(env.cost1[i]*x[i] for i in range(env.r)) + \
            gp.quicksum(gp.quicksum(gp.quicksum(env.cost2[s,i][j] * y[k][i,j] for i in range(env.r) for j in range(env.N))*env.prob[s] for s in tau[k]) for k in range(K) if tau[k])+\
            gp.quicksum(gp.quicksum(env.cost2[s,i][j] * y_s[s][i,j] for i in range(env.r) for j in range(env.N))*env.prob[s] for s in unseen_scen)
        spip.setObjective(obj_expr, GRB.MINIMIZE)

        con_assignment = {}
        con_capacity = {}
        # constraints
        for k in range(K) :
            for j in range(env.N):
                con_assignment[k,j] = spip.addConstr(gp.quicksum(y[k][i,j] for i in range(env.r))==1, name = f'con_assignment[{k},{j}]') 
            if len(tau[k]) :
                for s in tau[k] :
                    for i in range(env.r) :
                        con_capacity[k,s,i] = spip.addConstr(gp.quicksum(env.dem[s,i][j]*y[k][i,j] for j in range(env.N)) <= env.V[i]*x[i], name=f'con_capacity[{k},{s},{i}]') 

        con_assignment_s = {}
        con_capacity_s = {}
        for s in unseen_scen :
            for j in range(env.N) :
                con_assignment_s[s,j] = spip.addConstr(gp.quicksum(y_s[s][i,j] for i in range(env.r))==1, name = f'con_assignment_s[{s},{j}]') 
            for i in range(env.r) :
                con_capacity_s[s,i] = spip.addConstr(gp.quicksum(env.dem[s,i][j]*y_s[s][i,j] for j in range(env.N)) <= env.V[i]*x[i], name=f'con_capacity_s[{s},{i}]') 

        # solve
        spip.update()
        if spip._lazy_constraints == [] :
            spip.optimize()
        else :
            spip.optimize(self.lazy_callback)

        if not(spip.Status == gp.GRB.OPTIMAL):
            if spip.Status == gp.GRB.TIME_LIMIT : 
                status = 'time_limit'
            elif spip.Status == gp.GRB.INFEASIBLE : 
                status = 'infeasible'
            else :
                print(f'Error : Model status is {spip.Status}')
            obj_val = None
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
            obj_val_R = sum(env.cost1[i]*x_sol[i] for i in range(env.r))
            for k in range(K):
                y_sol[k] = {(i, j): var.X for (i, j), var in y[k].items()}
            for s in unseen_scen :
                y_s_sol[s] = {(i, j): var.X for (i, j), var in y_s[s].items()}
                obj_s = sum(env.cost2[s,i][j] * y_s_sol[s][i,j] for i in range(env.r) for j in range(env.N)) * env.prob[s]
                if obj_s > max_ob_s : 
                    max_ob_s = obj_s
                    s_max = s
            
        re_update = False

        if x_val is not None :
            re_update = True
            for i in range(env.r) : 
                x[i].LB = 0.0
                x[i].UB = 1.0

        if y_val is not None :
            re_update = True
            for k in range(K) :
                for i in range(env.r) :
                    for j in range(env.N): 
                        y[k][i,j].LB = 0.0
                        y[k][i,j].UB = 1.0

        if y_s_val is not None :
            re_update = True
            for s in unseen_scen:
                for i in range(env.r) :
                    for j in range(env.N): 
                        y_s[s][i,j].LB = 0.0
                        y_s[s][i,j].UB = 1.0
        
        if re_update :
            spip.update()
            
        return obj_val, x_sol, y_sol, y_s_sol, spip, obj_val_R, s_max, status
    
    def spip_update(self, env, spip, K, tau, k_new, s_new, x_val, y_val, y_s_val, leaf = False, skip = False, TL = None):
        # load variables
        x = {i: spip.getVarByName(f"x[{i}]") for i in range(env.r)}
        y = {k: {(i,j): spip.getVarByName(f"y[{k}][{i},{j}]") for j in range(env.N) for i in range(env.r)} for k in range(K)}

        list_scen   = [s for k in range(K) for s in tau[k]]
        unseen_scen = [s for s in range(env.N_scen) if s not in list_scen]

        y_s = {s: {(i,j): spip.getVarByName(f"y_s[{s}][{i},{j}]") for j in range(env.N) for i in range(env.r)} for s in unseen_scen}

        assert k_new is not None

        if not skip :
            if x_val is not None :
                for i in range(env.r) : 
                    x[i].LB = x_val[i]
                    x[i].UB = x_val[i]

            if y_val is not None :
                for k in range(K) :
                    if tau[k] :
                        if not k == k_new : 
                            for i in range(env.r) :
                                for j in range(env.N): 
                                    y[k][i,j].LB = y_val[k][i,j]
                                    y[k][i,j].UB = y_val[k][i,j]

            if y_s_val is not None :
                for s in unseen_scen :
                    for i in range(env.r) :
                        for j in range(env.N):
                            y_s[s][i,j].LB = y_s_val[s][i,j]
                            y_s[s][i,j].UB = y_s_val[s][i,j]

        for i in range(env.r) :
            for j in range(env.N):
                spip.remove(spip.getVarByName(f"y_s[{s_new}][{i},{j}]")) 
                y[k_new][i,j].Obj = y[k_new][i,j].Obj + env.cost2[s_new,i][j] * env.prob[s_new] 

        # add constraints :
        con_capacity = {}

        for i in range(env.r) :
            con_capacity[k_new,s_new,i] = spip.addConstr(gp.quicksum(env.dem[s_new,i][j]*y[k_new][i,j] for j in range(env.N)) <= env.V[i]*x[i], name=f'con_capacity[{k_new},{s_new},{i}]') 

        temp_con_to_remove_s = []
        for j in range(env.N):
            temp_con_to_remove_s.append(spip.getConstrByName(f"con_assignment_s[{s_new},{j}]"))
        for i in range(env.r):
            temp_con_to_remove_s.append(spip.getConstrByName(f"con_capacity_s[{s_new},{i}]"))

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
            obj_val_R = sum(env.cost1[i]*x_val[i] for i in range(env.r)) 
            obj_val = obj_val_R + sum(sum(sum(env.cost2[s,i][j] * y_val[k][i,j] for i in range(env.r) for j in range(env.N)) * env.prob[s]
                                                    for s in tau[k]) for k in range(K) if tau[k])
            # scenario selection
            max_ob_s = - np.inf
            s_max = env.N_scen
            for s in unseen_scen :
                obj_s = sum(env.cost2[s,i][j] * y_s_val[s][i,j] for i in range(env.r) for j in range(env.N)) * env.prob[s]
                obj_val += obj_s
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
                y_s_sol  = {}
                max_ob_s = - np.inf
                s_max = env.N_scen
                obj_val_R = sum(env.cost1[i]*x_sol[i] for i in range(env.r))
                for k in range(K):
                    y_sol[k]   = {(i, j): var.X for (i, j), var in y[k].items()}
                for s in unseen_scen :
                    y_s_sol[s] = {(i, j): var.X for (i, j), var in y_s[s].items()}
                    obj_s = sum(env.cost2[s,i][j] * y_s_sol[s][i,j] for i in range(env.r) for j in range(env.N)) * env.prob[s]
                    if obj_s > max_ob_s : 
                        max_ob_s = obj_s
                        s_max = s

            re_update = False

            if x_val is not None :
                re_update = True
                for i in range(env.r) : 
                    x[i].LB = 0.0
                    x[i].UB = 1.0

            if y_val is not None :
                re_update = True
                for k in range(K) :
                    for i in range(env.r) :
                        for j in range(env.N) : 
                            y[k][i,j].LB = 0.0
                            y[k][i,j].UB = 1.0

            if y_s_val is not None :
                re_update = True
                for s in unseen_scen:
                    for i in range(env.r) :
                        for j in range(env.N) : 
                            y_s[s][i,j].LB = 0.0
                            y_s[s][i,j].UB = 1.0

            if leaf :
                if k_new is not None :
                    for i in range(env.r):
                        for j in range(env.N):
                            y[k_new][i,j].Obj = y[k_new][i,j].Obj - env.cost2[s_new,i][j] * env.prob[s_new]
                    y_s = {}
                    y_s[s_new] = spip.addVars(env.r,env.N, vtype=GRB.BINARY, name=f"y_s[{s_new}]" )
                    
                    # constraints
                    for j in range(env.N) :
                        spip.addConstr(gp.quicksum(y_s[s_new][i,j] for i in range(env.r))==1, name = f'con_assignment_s[{s_new},{j}]') 
                    for i in range(env.r) :
                        spip.addConstr(gp.quicksum(env.dem[s_new,i][j]*y_s[s_new][i,j] for j in range(env.N)) <= env.V[i]*x[i], name=f'con_capacity_s[{s_new},{i}]') 

                    # remove constraints :
                    for i in range(env.r) :
                        spip.remove(con_capacity[k_new,s_new,i])

                    re_update = True 
                s_max = s_new
            
            if re_update :
                spip.update()
                
            return obj_val, x_sol, y_sol, y_s_sol, spip, obj_val_R, s_max, status 

    def spip_resolve(self, env, K, tau, spip, TL = None):
        x = {i: spip.getVarByName(f"x[{i}]") for i in range(env.r)}
        y = {k: {(i,j): spip.getVarByName(f"y[{k}][{i},{j}]") for j in range(env.N) for i in range(env.r)} for k in range(K)}

        list_scen   = [s for k in range(K) for s in tau[k]]
        unseen_scen = [s for s in range(env.N_scen) if s not in list_scen]

        y_s = {s: {(i,j): spip.getVarByName(f"y_s[{s}][{i},{j}]") for j in range(env.N) for i in range(env.r)} for s in unseen_scen}

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
            x_sol = {i: spip.getVarByName(f"x[{i}]").X for i in range(env.r)}
            y_sol = {k: {(i,j): spip.getVarByName(f"y[{k}][{i},{j}]").X for i in range(env.r) for j in range(env.N)} for k in range(K)}
            y_s_sol = {s : {(i,j): spip.getVarByName(f"y_s[{s}][{i},{j}]").X for i in range(env.r) for j in range(env.N)} for s in unseen_scen}
            max_ob_s = - np.inf
            s_max = env.N_scen
            for s in unseen_scen:
                obj_s = sum(env.cost2[s,i][j] * y_s_sol[s][i,j] for i in range(env.r) for j in range(env.N)) * env.prob[s]
                if obj_s > max_ob_s : 
                    max_ob_s = obj_s
                    s_max = s
            obj_val_R = sum(env.cost1[i]*x_sol[i] for i in range(env.r))
            
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
        x = {i: model.getVarByName(f"x[{i}]") for i in range(env.r)}
        y = {k: {(i,j) : model.getVarByName(f"y[{k}][{i},{j}]") for j in range(env.N) for i in range(env.r)} for k in range(K)}

        Sx = [i for i in range(env.r) if x_sol[i] >= 1 - 1e-6]
        Tx = [i for i in range(env.r) if x_sol[i] <= 1e-6]

        LSx = len(Sx)

        Sy = {}
        Ty = {}

        LSy = 0

        for k in range(K):
            for i in range(env.r):
                Sy[k,i] = [j for j in range(env.N) if y_sol[k][i,j] >= 1 - 1e-6]
                Ty[k,i] = [j for j in range(env.N) if y_sol[k][i,j] <= 1e-6]
                LSy += len(Sy[k,i])

        lhs = gp.LinExpr()

        for i in range(env.r): 
            if i in Sx:
                lhs += x[i]
            else:
                lhs -= x[i]

        for k in range(K):
            for i in range(env.r):
                for j in range(env.N):
                    if j in Sy[k,i]:
                        lhs += y[k][i,j]
                    else :
                        lhs -= y[k][i,j]

        rhs = LSx + LSy - 1

        return lhs, rhs

    def sip(self, env, K, x, y, s):
        Klist = [k for k in range(K) 
                    if all(sum(env.dem[s,i][j]*y[k][i,j] for j in range(env.N)) <= env.V[i]*x[i] for i in range(env.r))]

        if Klist == [] : 
            return False, None, None
        else :
            best_ob = np.inf
            j = K # outside of real scenario range
            for k in Klist :
                obk = sum(env.cost2[s,i][j] *y[k][i,j] for i in range(env.r) for j in range(env.N)) * env.prob[s]
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
        for k in range(K):
            y[k] = compact.addVars(env.r,env.N, vtype=GRB.BINARY, name=f"y[{k}]")
            for s in range(env.N_scen):
                u[s,k] = compact.addVar(vtype=GRB.BINARY, name = f"u[{s},{k}]")

        # objective function
        obj_expr = gp.quicksum(env.cost1[i]*x[i] for i in range(env.r)) +  gp.quicksum( gp.quicksum(env.cost2[s,i][j] * gp.quicksum(y[k][i,j]*u[s,k] for k in range(K)) for i in range(env.r) for j in range(env.N))*env.prob[s] for s in range(env.N_scen))
        compact.setObjective(obj_expr, GRB.MINIMIZE)

        # constraints
        for s in range(env.N_scen):
            for k in range(K):
                for i in range(env.r):
                    compact.addGenConstrIndicator(u[s,k], True, gp.quicksum(env.dem[s,i][j]*y[k][i,j] for j in range(env.N)) <= env.V[i]*x[i] )
        for k in range(K):
            for j in range(env.N):
                compact.addConstr(gp.quicksum(y[k][i,j] for i in range(env.r))== 1)

        compact.addConstrs(gp.quicksum(u[s,k] for k in range(K)) == 1 for s in range(env.N_scen))

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
            for k in range(K) : 
                y_sol[k] = {(i, j): var.X for (i, j), var in y[k].items()}
            for k in range(K):
                u_sol[k] = [s  for s in range(env.N_scen) if u[s,k].X==1]
            
        return feas, obj_val, x_sol, y_sol, u_sol, self.inc_UB

    def my_callback(self, model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # Time (in seconds since start)
            time = model.cbGet(gp.GRB.Callback.RUNTIME)
            # Best incumbent objective (upper bound)
            ub = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            
            # Save if not already saved at that time
            self.inc_UB[time] = ub