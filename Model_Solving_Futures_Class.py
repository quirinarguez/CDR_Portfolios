# Importing Relevant Libraries
import os, os.path
import importlib
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
import pyDOE2 as pyd
import seaborn as sns
from matplotlib.pyplot import plot, savefig
import time
import datetime 
from datetime import date
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.ticker as mtick
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import logging
from pyomo.environ import *

# Source: https://danshiebler.com/2016-09-14-parallel-progress-bar/
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys # Python "sys" documentation: https://docs.python.org/3/library/sys.html

# To reset the default parameters of matplotlib:
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


# ======= DEFINING RELEVANT FUNCTIONS =======
# ===================================== Saving results from optimisation =====================================    
def key_parameters():
    # Useful non-varying parameters
    i = ['BECCS', 'A/R', 'SCS', 'BC', 'DACCS', 'EW', 'OA', 'DOCCS']
    k1 = 0 # Value of first time period
    delK = 5 # Time step of optimisation (i.e., length of each period)
    N = 4 # Number of segments for piecewise linear approximation
    first_year = 2020 # Year corresponding to first time period
    last_year = 2100 # Year corresponding to last time period
    T = int(1 + (last_year-first_year)/delK) # Number of time periods evaluated, equivalent to 17 if delK = 5 (i.e., 80 years)
    k = list(range(k1,T))

    return i, k, k1, delK, N, first_year, last_year, T

def extract_results(Instance_bin,results_y,results_x,results_b,results_z,results_Costs,results_C,results_E,results_L,results_obj):
    results_y += [Instance_bin.y.extract_values()]
    results_x += [Instance_bin.x.extract_values()]
    results_b += [Instance_bin.b.extract_values()]
    results_z += [Instance_bin.z.extract_values()]
    results_Costs += [Instance_bin.costs.extract_values()]
    results_C += [Instance_bin.c.extract_values()]
    results_E += [Instance_bin.e.extract_values()]
    results_L += [Instance_bin.ld.extract_values()]
    results_obj += [Instance_bin.OBJ()]
    return results_y,results_x,results_b,results_z,results_Costs,results_C,results_E,results_L,results_obj

def null_results(Instance_bin,results_y,results_x,results_b,results_z,results_Costs,results_C,results_E,results_L,results_obj):
    i, k, k1, delK, N, first_year, last_year, T = key_parameters()
    merged_list = [(i[n], k[m]) for n in range(0, len(i)) for m in range(k1, len(k))]
    
    results_y += [dict.fromkeys(merged_list, np.NaN)]
    results_x += [dict.fromkeys(merged_list, np.NaN)]
    results_b += [dict.fromkeys(merged_list, np.NaN)]
    results_z += [dict.fromkeys(merged_list, np.NaN)]
    results_Costs += [dict.fromkeys(merged_list, np.NaN)]
    results_C += [dict.fromkeys(merged_list, np.NaN)]
    results_E += [dict.fromkeys(k, np.NaN)]
    results_L += [dict.fromkeys(k, np.NaN)]
    results_obj += [np.NaN]
    return results_y, results_x, results_b, results_z, results_Costs, results_C,results_E,results_L,results_obj

def unit_costs_b(y, b, Y_ref, SC_ref):
    return SC_ref*((y/Y_ref)**(-b))

def abstract_model():
# ======================================================= Input Parameters =======================================================
    i, k, k1, delK, N, first_year, last_year, T = key_parameters()
    '''k1 = 0 # Value of first time period
    delK = 5 # Time step of optimisation (i.e., length of each period)
    N = 4 # Number of segments for piecewise linear approximation
    first_year = 2020 # Year corresponding to first time period
    last_year = 2100 # Year corresponding to last time period
    T = int(1 + (last_year-first_year)/delK) # Number of time periods evaluated, equivalent to 17 if delK = 5 (i.e., 80 years)'''
    
    # ======================================================= Model Creation =======================================================
    m = AbstractModel()

    # ------------------ Indices ------------------ 
    m.i = Set(initialize=['BECCS', 'A/R', 'SCS', 'BC', 'DACCS', 'EW', 'OA', 'DOCCS']) # CDR Options (=8)
    m.k = Set(initialize=list(range(k1,T))) # Time period, initialised to 10 time periods
    m.l = Set(initialize=list(range(1,N+1))) # State of Endogenous Learning: line segments (=4) for piecewise linear function

    # ------------------ Parameters ------------------ 
    # Binary Parameters
    m.S = Param(m.i, initialize={'BECCS': 1, 'A/R':0, 'SCS':0, 'BC':0, 'DACCS':1, 'EW':0, 'OA':0, 'DOCCS':1})
        # Storage requirement per NET i: only 1 for BECCS, DACCS & DOCCS (binary parameter)
    m.T = Param(m.i,m.k) # CDR option $i$ is available (0) or not (1) at time $k$ (binary parameter)

    # Non-indexed parameters
    m.rE = Param(default=0.03) # Economic Discount Rate

    m.delK = Param(default=delK) # Time step or length of each time interval

    m.Q = Param(m.k) #Geological Storage Limit per year
    m.G = Param() #Cumulative negative emission target by 2100

    # Characterising CDR potentials
    m.Xmax = Param(m.i,m.k) # Limit annual negative emission potential per CDR option per period
    m.Yini = Param(m.i) # Initial values for removals and installed capacity at k=0
    m.Ymax = Param(m.i) # Limit cumulative negative emission potential per CDR option per period

    # Implementing Plant Lifetime
        # Lifetime of initial capacity of CDR option $i$ (years)
    m.LTIni = Param(m.i,initialize={'BECCS': 10, 'A/R':25, 'SCS':5, 'BC':10, 'DACCS':10, 'EW':5, 'OA':5, 'DOCCS':5}) # Default Scenario
    #m.LTIni = Param(m.i,initialize={'BECCS': 5, 'A/R':5, 'SCS':5, 'BC':5, 'DACCS':5, 'EW':5, 'OA':5, 'DOCCS':5}) # Sensitivity Analysis: Scenario C
    #m.LTIni = Param(m.i,initialize={'BECCS': 20, 'A/R':55, 'SCS':5, 'BC':25, 'DACCS':25, 'EW':10, 'OA':5, 'DOCCS':5})  # Sensitivity Analysis: Scenario D
        
        # Lifetime of CDR option $i$ (years)
    m.LT = Param(m.i,initialize={'BECCS': 20, 'A/R':55, 'SCS':20, 'BC':25, 'DACCS':25, 'EW':10, 'OA':20, 'DOCCS':25}) 

    # Characterising resource consumption and limits
    m.LandL = Param(m.k) # Land Limit
    m.EnergyL = Param(m.k) # Energy Limit
    m.LandR = Param(m.i,m.k) # Land Used
    m.EnergyR = Param(m.i,m.k) # Energy Used
    m.OPEX = Param(m.i,m.k) # Operational costs for running CDR option i in period k

    # Upper and Lower boundaries of Piecewise linear COST formulation of the learning curve model
    m.yLo = Param(m.i,m.l) # lower segment x-value of cumulative capacity of CDR option i of piecewise linear cost function
    m.yUp = Param(m.i,m.l) # Upper segment x-value of cumulative capacity of CDR option i of piecewise linear cost function
    m.CLo = Param(m.i,m.l) # Value for lower bound of cost of CDR option i at segment l
    m.CUp = Param(m.i,m.l) # Value for upper bound of cost of CDR option i at segment l
    m.Beta = Param(m.i,m.l) # Slope of piecewise linear cost function (calculated in Excel spreadsheet)

    # ------------------ Variables ------------------ 
    m.x = Var(m.i, m.k, domain=NonNegativeReals) # tCO2 of installed capacity of CDR option i in time period k (tCO2/period)
    m.b = Var(m.i, m.k, domain=NonNegativeReals) # tCO2 removed from CDR option i in time period k (tCO2)
    m.y = Var(m.i, m.k, domain=NonNegativeReals) # tCO2 of capacity of CDR option i installed at time period k (tCO2)
    m.z = Var(m.i, m.k) # tCO2 of cumulative installed capacity of CDR option i at time period k (tCO2)

    m.lamb = Var(m.i, m.k, m.l, domain=NonNegativeReals) # position for CDR option i in time period k 
    m.delta = Var(m.i,m.k,m.l, domain=Binary) # CDR option i at time k is in state l (d_ikl = 1) or not (d_ikl = 0) (binary parameter)

    m.c = Var(m.i, m.k, domain=NonNegativeReals) # Endogenous cumulative cost of CDR option i at time period k
    m.costs = Var(m.i, m.k, domain=NonNegativeReals) # Cost of CDR option i in time period k

    m.e = Var(m.k)#, domain=NonNegativeReals) # Energy used
    m.ld = Var(m.k, domain=NonNegativeReals) # Land used

    # ------------------ Objective ------------------ 
    def costs_in_period(m,I,K):
        if K == m.k.first():
            return m.costs[I,K] == m.c[I,K] + m.b[I,K]*m.OPEX[I,K] # Costs for first period are equal to cumulative costs (0 at first period) + OPEX*removals
        else:
            return m.costs[I,K] == m.b[I,K]*m.OPEX[I,K] + (m.c[I,K] - m.c[I,K-1]) # Costs for period k are (tCO2removed in k)*OPEX + (cumulative costs in k) - (cumulative costs in previous k)
    m.Costs_in_period = Constraint(m.i,m.k, rule=costs_in_period)

    def obj_expression(m):
        return sum(m.costs[I,K]*((1+m.rE)**(-K*m.delK)) for I in m.i for K in m.k)
    m.OBJ = Objective(rule=obj_expression, sense=minimize)

    # ------------------ Constraints ------------------ 
    # ======================= Flow Equilibrium Constraints =======================
    # Set initial x to initial cumulative capacity, and define maximum capacity deployment per period.
    def cont_rule_x(m,I,K):
        if K == m.k.first():
            return m.x[I,K] == m.Yini[I]
        else:
            return m.x[I,K] <= m.Xmax[I,K]*m.delK*m.T[I,K]
    m.Cont_x = Constraint(m.i,m.k, rule=cont_rule_x)

    # Continuity of variable y
    def cont_rule_y(m,I,K):
        if K == m.k.first():
            return m.y[I,K] == m.Yini[I]
        elif K < (k1 + m.LTIni[I]/m.delK): 
            return m.y[I,K] == m.y[I,K-1] + m.x[I,K]
        elif K == (k1 + m.LTIni[I]/m.delK): 
            return m.y[I,K] == m.y[I,K-1] + m.x[I,K] - m.x[I,K-m.LTIni[I]/m.delK]
        elif (K > (k1 + m.LTIni[I]/m.delK)) and (K <= (k1 + m.LT[I]/m.delK)):
            return m.y[I,K] == m.y[I,K-1] + m.x[I,K]
        else:
            return m.y[I,K] == m.y[I,K-1] + m.x[I,K] - m.x[I,K-m.LT[I]/m.delK]
    m.Cont_y = Constraint(m.i,m.k, rule=cont_rule_y)
        
    # Continuity of variable b
    def cont_rule_b(m,I,K):
        if K == m.k.first():
            return m.b[I,K] == m.Yini[I]*m.delK
        else:
            return m.b[I,K] == m.y[I,K-1]*m.delK # Note the *m.delK: this refers to b being in "per period (tCO2)" and x & y being  in "per year (tCO2/yr)"
    m.Cont_b = Constraint(m.i,m.k, rule=cont_rule_b)

    # Continuity of variable z
    def cont_rule_z(m,I,K):
        return m.z[I,K] == sum(m.x[I,KK] for KK in m.k if KK <= K)
    m.Cont_z = Constraint(m.i,m.k, rule=cont_rule_z)

    # ======================= Constraints of Endogenous Variable(s) =======================
    # Continuity of binary variable delta
    def bin_delta(m,I,K):
        return sum(m.delta[I,K,L] for L in m.l) == 1
    m.Bin_delta = Constraint(m.i,m.k, rule=bin_delta)

    # Sum of sub-components of y
    def Sub_y(m,I,K):
        return sum(m.lamb[I,K,L] for L in m.l) == sum(m.x[I,KK] for KK in m.k if KK <= K) 
    m.sub_dy = Constraint(m.i,m.k, rule=Sub_y)

    # Defining Lower bound of Lambda
    def lower_bound_lamb(m,I,K,L):
        return m.lamb[I,K,L] >= m.yLo[I,L]*m.delta[I,K,L]
    m.Lower_bound_lamb = Constraint(m.i,m.k,m.l, rule=lower_bound_lamb)

    # Defining Upper bound of Lambda
    def upper_bound_lamb(m,I,K,L):
        return m.lamb[I,K,L] <= m.yUp[I,L]*m.delta[I,K,L]
    m.Upper_bound_lamb = Constraint(m.i,m.k,m.l, rule=upper_bound_lamb)

    # Defining Cumulative Endogenous Costs
    def cum_costs(m,I,K):
        return m.c[I,K] == sum(m.CLo[I,L]*m.delta[I,K,L] + m.Beta[I,L]*(m.lamb[I,K,L]-m.yLo[I,L]*m.delta[I,K,L]) for L in m.l)
    m.Cum_costs = Constraint(m.i,m.k, rule=cum_costs)

    # ======================= Resource Constraints =======================
    # Meeting Neg Emissions
    def NE_rule(m):
        return sum(m.b[I,K] for I in m.i for K in m.k) >= m.G # The *m.delK is now in the definition of b :)
    m.NE = Constraint(rule=NE_rule)

    # Defining Land Used
    def land_used(m,K):
        #return m.ld[K] == sum((m.y[I,K]-m.Yini[I])*m.LandR[I,K] for I in m.i)
        return m.ld[K] == sum(m.y[I,K]*m.LandR[I,K] for I in m.i)
    m.Land_used = Constraint(m.k, rule=land_used)

    # Land Env Footprints
    def land_rule(m,K):
        return sum(m.y[I,K]*m.LandR[I,K] for I in m.i) <= m.LandL[K] 
        #return sum((m.y[I,K]-m.Yini[I])*m.LandR[I,K] for I in m.i) <= m.LandL[K] 
    m.Land = Constraint(m.k, rule=land_rule)

    # Defining Energy Used
    def energy_used(m,K):
        return m.e[K] == sum(m.b[I,K]*m.EnergyR[I,K] for I in m.i)
        #return m.e[K] == sum((m.b[I,K]-m.Yini[I]*m.delK)*m.EnergyR[I,K] for I in m.i)
    m.Energy_used = Constraint(m.k, rule=energy_used)

    # Energy Env Footprints
    def energy_rule(m,K):
        #return m.e[K] <= m.EnergyL[K]     
        return sum((m.b[I,K]-m.Yini[I]*m.delK)*m.EnergyR[I,K] for I in m.i) <= m.EnergyL[K] 
    m.Energy = Constraint(m.k, rule=energy_rule)

    # Geological storage
    def stor_rule(m,K):
        return sum(m.y[I,K] * m.S[I] for I in m.i) <= m.Q[K] 
    m.Stor = Constraint(m.k, rule=stor_rule)

    # Below Maximum cumulative potential for each CDR Option
    def max_y_rule(m,I,K):
        return m.y[I,K] <= m.Ymax[I]
    m.Max_y = Constraint(m.i,m.k, rule=max_y_rule)

    return m

def solving_future(info):
    futureID = info[0]
    dict_ID = info[1]
    
    ID_time = time.perf_counter()
    wanted = False # True, for printing the Experience curves for all CDR options for EACH future ID evaluated
    Tee = False # The optional keyword argument tee=True causes a more verbose display of solver log outputs
    
    i, k, k1, delK, N, first_year, last_year, T = key_parameters()
    merged_list = [(i[n], k[m]) for n in range(0, len(i)) for m in range(k1, len(k))]

    # Files to save and return results
    results_to_save = ['y','x','b','z','costs','c','e','ld','OBJ']
    null_results_to_save = {'y': dict.fromkeys(merged_list, np.NaN),
                       'x':dict.fromkeys(merged_list, np.NaN),
                       'b':dict.fromkeys(merged_list, np.NaN),
                       'z':dict.fromkeys(merged_list, np.NaN),
                       'costs':dict.fromkeys(merged_list, np.NaN),
                       'c':dict.fromkeys(merged_list, np.NaN),
                       'e':dict.fromkeys(k, np.NaN),
                       'ld':dict.fromkeys(k, np.NaN),
                       'OBJ':np.NaN}
    
    results_to_return = {}
    results = []
    # -------------------------------- Instantiate the model for the current future ID data --------------------------------
    data = {}
    for key in dict_ID.keys():
        data[key] = dict_ID[key]
    
    m = abstract_model()
    #Instance_bin = m.create_instance(data)
    Instance_bin = m.create_instance({None:data})

    # ======================= Scaling Factors ======================= 
    # See Pyomo documentation for more info: https://pyomo.readthedocs.io/en/stable/model_transformations/scaling.html
    # Create the scaling factors
    Instance_bin.scaling_factor = Suffix(direction=Suffix.EXPORT)

    Instance_bin.scaling_factor[Instance_bin.x] = 1e-6    # scale the x variable
    Instance_bin.scaling_factor[Instance_bin.b] = 1e-6    # scale the y variable
    Instance_bin.scaling_factor[Instance_bin.y] = 1e-6    # scale the b variable
    Instance_bin.scaling_factor[Instance_bin.z] = 1e-6    # scale the z variable
    Instance_bin.scaling_factor[Instance_bin.lamb] = 1e-6    # scale the lambda variable
    Instance_bin.scaling_factor[Instance_bin.c] = 1e-9    # scale the c variable
    Instance_bin.scaling_factor[Instance_bin.costs] = 1e-6    # scale the costs variable

    scaled_model = TransformationFactory('core.scale_model').create_using(Instance_bin)

    # -------------------------------- Solve the model for the current future ID data --------------------------------
    opt = SolverFactory('gurobi')
    opt.options['mipgap'] = 0.02 
        # The MIP solver will terminate (with an optimal result) when the gap between the lower and upper objective bound is less than MIPGap times the absolute value of the incumbent objective value. 
    #opt.options['TimeLimit'] = 20 # Potential time limit per iteration (Default = disabled)
    #opt.options['NoRelHeurWork'] = 1 # This option is useful when the solver gets 'blocked' (i.e., time elapsed > 5mins) in finding a solution (Default = disabled)

    log_bin = opt.solve(scaled_model, options={"CliqueCuts":0}, tee=Tee) # The optional keyword argument tee=True causes a more verbose display of solver log outputs."Cuts":1,
    #return log_bin

    # ------------- Retrieving solver results and saving them to a "results_" list if solution is found ------------- 
    # If solution is feasible and optimal:
    if (log_bin.solver.status == SolverStatus.ok) and (log_bin.solver.termination_condition == TerminationCondition.optimal):
        TransformationFactory('core.scale_model').propagate_solution(scaled_model,Instance_bin)
        Instance_bin.solutions.load_from(log_bin)
        print('Optimal solution for future ID %d found after %.1f seconds.' % (futureID, (time.perf_counter() - ID_time)))   
        #solved_id += [futureID]
        #solved_type += ["mip"]
        #extract_results(Instance_bin,results_y,results_x,results_b,results_z,results_Costs,results_C,results_E,results_L,results_obj)
        SAVING_results_to_save = {'y': Instance_bin.y.extract_values(),
                       'x':Instance_bin.x.extract_values(),
                       'b':Instance_bin.b.extract_values(),
                       'z':Instance_bin.z.extract_values(),
                       'costs':Instance_bin.costs.extract_values(),
                       'c':Instance_bin.c.extract_values(),
                       'e':Instance_bin.e.extract_values(),
                       'ld':Instance_bin.ld.extract_values(),
                       'OBJ':Instance_bin.OBJ()}
        
        solved = 1
        for result in results_to_save:
            results_to_return[result] = [SAVING_results_to_save[result]]
    else:
        print('Solution for future ID %d NOT found. Solver status is %s. Termination condition is %s (%.1f seconds elapsed).' % (futureID, log_bin.solver.status, log_bin.solver.termination_condition, (time.perf_counter() - ID_time)))       
        #solved += -1
        #solved_binary.update({futureID:0})
        #solved_type += ["none"]
        #null_results(Instance_bin,results_y,results_x,results_b,results_z,results_Costs,results_C,results_E,results_L,results_obj)
        for result in results_to_save:
            results_to_return[result] = [null_results_to_save[result]]
        solved = 0
    if Tee:
        print()
    #time.sleep(1)
    
    results += [futureID, results_to_return, solved]
    
    return results

def main():
    print('Reading main :)')
    # ------------------ Parameters not imported/computed ------------------
    i, k, k1, delK, N, first_year, last_year, T = key_parameters()

    '''# ## Defining parameters that do not change across futures
    k1 = 0 # Value of first time period
    delK = 5 # Time step of optimisation (i.e., length of each period)
    N = 4 # Number of segments for piecewise linear approximation
    first_year = 2020 # Year corresponding to first time period
    last_year = 2100 # Year corresponding to last time period
    T = int(1 + (last_year-first_year)/delK) # Number of time periods evaluated, equivalent to 17 if delK = 5 (i.e., 80 years)

    i = ['BECCS', 'A/R', 'SCS', 'BC', 'DACCS', 'EW', 'OA', 'DOCCS']
    k = list(range(k1,T))'''

    # Number of futures to be evaluated
    n_lhs = 3000 

    # Color Map for later plotting
    cmap=plt.get_cmap('Set2')
    color_dict = {'BECCS':cmap(0.8), 'A/R':cmap(0.1), 'SCS':cmap(0.9), 'BC':cmap(0.2), 'DACCS':cmap(0.7), 'EW':cmap(0.3), 'OA':cmap(0.4), 'DOCCS':cmap(0.5)}

    # i-k Indexed Parameters
    merged_list = [(i[n], k[m]) for n in range(0, len(i)) for m in range(k1, len(k))]
    t_ik = dict.fromkeys(merged_list, 1)
    t_ik.update({('DACCS',n): 0 for n in range(k1+1,int((2030-first_year)/delK))}) # i.e. available from 2030
    t_ik.update({('DOCCS',n): 0 for n in range(k1+1,int((2040-first_year)/delK))}) # i.e. available from 2040
    t_ik.update({('OA',n): 0 for n in range(k1+1,int((2050-first_year)/delK))}) # i.e. available from 2050

    active_k = dict.fromkeys(i, 0)
    for CDR in i:
        for K in k:
            if t_ik[CDR,K] == 1:
                active_k[CDR] = active_k.get(CDR) + 1
                
    # i-l Indexed Parameters
    l = list(range(1,N+1))
    merged_il_list = [(i[n], l[m]) for n in range(0, len(i)) for m in range(0, len(l))]
    il = ['Yref', 'Ymax', 'ER', 'SCref']
    Fac = [1 / (2 ** (N - L)) for L in l]
    #Fac = [1 / (2 ** (N - L)) for L in list(range(0, N))]

    j = ['Land', 'Energy', 'OPEX','Xmax'] # Add here CDR specific parameters not indexed by l

    # Non-CDR Parameters
    non_i = ['CDRRequired','LandL','EnergyL','GeoSt'] # Add here non-CDR-specific parameters
    merged_ikl_list = [(i[n], k[m], l[o]) for n in range(0, len(i)) for m in range(k1, len(k)) for o in range(0,len(l))]


    # ## Creating a Latin Hypercube from Uncertain Parameters
    # 
    # To create a Latin Hypercube, one from two sampling strategies can be selected based on:
    # - The sampling step 
    # - The number of possible values for each parameter

    # ======= WITH SAMPLING STEP =======
    # 
    # Note this uses a DIFFERENT Excel file
    
    # ===================================== Implementing Sampling with data input =====================================
    df = pd.read_excel("PortfolioFiles/Portfolio_Files_Latest/All data Step size.xlsx", index_col = "Parameter")

    units = dict(zip(df.index, df.Unit)) # Save all units in a dictionary for later use

    # Separate all constant parameters
    constant = df[df.minimum == df.maximum]
    constant = constant[['minimum']].T
    constant = pd.concat([constant]*(n_lhs+1),axis=0,ignore_index=True)

    # Get a data frame with selected columns
    df = df[df.minimum != df.maximum]

    # Modify dataframe by Sampling step
    df_Sampling_step = df[['Sampling_step']]
    df = df[['minimum','maximum']]
    df = df.div(df_Sampling_step.Sampling_step, axis=0)

    # =====================================
    # Implementing Sampling with data input
    # =====================================
    dict_ranges = df.T.to_dict('records')
    dict_ranges = dict_ranges[0]

    lbounds = df["minimum"].values.tolist()
    lbounds = [int(x) for x in lbounds]

    ubounds  = df["maximum"].values.tolist()
    ubounds = [int(x) for x in ubounds]

    sampler = qmc.LatinHypercube(d=len(lbounds))
    sample = sampler.integers(l_bounds=lbounds, u_bounds=ubounds, n=n_lhs, endpoint=True)

    df_LHS = pd.DataFrame.from_records(sample)    
    df_LHS = df_LHS.set_axis(dict_ranges, axis=1)
    df_LHS = pd.concat([df[["minimum"]].T,df_LHS], axis = 0, ignore_index=True) # Add the min values for "future ID_0"
    df_LHS.index.names = ['future_id']

    # Return to original units
    df_LHS = df_LHS.T.mul(df_Sampling_step.Sampling_step, axis=0)

    # Creating model input table
    df_input = pd.concat([df_LHS.T,constant], axis=1)
    df_input.index.names = ['future_id']
    df_input.head()


    '''# ======= WITH SAMPLING SIZE =======

    # ===================================== Implementing Sampling with data input =====================================
    df = pd.read_excel("PortfolioFiles/Portfolio_Files_Latest/All data.xlsx", index_col = "Parameter")

    units = dict(zip(df.index, df.Unit)) # Save all units in a dictionary for later use

    # Separate all constant parameters
    constant = df[df.minimum == df.maximum]
    constant = constant[['minimum']].T
    constant = pd.concat([constant]*(n_lhs+1),axis=0,ignore_index=True)

    # Get a data frame with selected columns
    df = df[df.minimum != df.maximum]

    # Modify dataframe by Sampling step
    df_Sampling_step = df[['Sampling_step']]
    df_Number_samples = df[['N_samples']].rename(columns={'N_samples':"maximum"})
    df_Number_samples[["minimum"]] = 1
    df = df[['minimum','maximum']]

    # =====================================
    # Implementing Sampling with data input
    # =====================================
    dict_ranges = df.T.to_dict('records')[0]

    lbounds = df_Number_samples["minimum"].values.tolist()
    lbounds = [int(x) for x in lbounds]

    ubounds  = df_Number_samples["maximum"].values.tolist()
    ubounds = [int(x) for x in ubounds]

    sampler = qmc.LatinHypercube(d=len(lbounds))
    sample = sampler.integers(l_bounds=lbounds, u_bounds=ubounds, n=n_lhs, endpoint=True)

    df_LHS = pd.DataFrame.from_records(sample)    
    df_LHS = df_LHS.set_axis(dict_ranges, axis=1)
    df_LHS = pd.concat([df[["minimum"]].T,df_LHS], axis = 0, ignore_index=True) # Add the min values for "future ID_0"
    df_LHS.index.names = ['future_id']

    # Return to original units
    #df_LHS = df_LHS.T.mul(df_Sampling_step.Sampling_step, axis=0)
    for futureID in range(1,n_lhs+1):
        for parameter in df_LHS.columns:
            if df_LHS.loc[futureID,parameter] == 1:
                df_LHS.loc[futureID,parameter] = df.loc[parameter,"minimum"]
            else:
                df_LHS.loc[futureID,parameter] = round(df.loc[parameter,"minimum"] + (df_LHS.loc[futureID,parameter]-1)*df_Sampling_step.loc[parameter,"Sampling_step"], 3)

    # Creating model input table
    df_input = pd.concat([df_LHS,constant], axis=1)
    df_input.index.names = ['future_id']
    df_LHS = df_LHS.T
    df_input.head()'''

    Uncertainties = dict(zip(df_LHS.index.to_list(),['DACCS Energy Requirements','DOCCS Energy Requirements','EW Energy Requirements', 'OA Energy Requirements', 'Year at which energy becomes available for CDR',
    'A/R Land Requirements','BECCS Land Requirements','Land Limit','A/R Experience Parameter','BC Experience Parameter','BECCS Experience Parameter','DACCS Experience Parameter',
    'DOCCS Experience Parameter','EW Experience Parameter','OA Experience Parameter','SCS Experience Parameter','Removals Required (2020 - 2100)','A/R Maximum Potential', 'BC Maximum Potential', 'BECCS Maximum Potential',
    'DACCS Maximum Potential', 'DOCCS Maximum Potential', 'EW Maximum Potential', 'OA Maximum Potential', 'SCS Maximum Potential', 'Annual Limit to Geological Injection']))
    

    # ======= IMPORTING MODELLING INPUTS =======

    # If the kernel is restarted: run any of the "With sampling step / size" cells and then run this one
    #today = date.today().strftime("%d.%m.%Y")
    today = "15.08.2024"

    df_input = pd.read_excel("Input_Output_Files/File Name_%s.xlsx" %today, index_col ='future_id')

    # If only the LHS table is wanted:
    df_LHS = df_input[list(dict_ranges.keys())].T

    # If only the Y_ref is wanted:
    Initial_YRef = df_input[df_input.columns.intersection(['Yref_' + el for el in i])]
    Initial_YRef.columns = [str(col).split('_')[1] for col in Initial_YRef.columns]
    Initial_YRef = Initial_YRef.loc[0].to_dict()

    # ======= PREPARING THE DATA INPUT FOR EACH FUTURE =======

    # ### Gurobi
    # 
    # - [Parameters Documentation](https://www.gurobi.com/documentation/current/refman/parameters.htmlhttps://www.gurobi.com/documentation/current/refman/parameters.html)
    #     - [Cuts](https://www.gurobi.com/documentation/9.1/refman/cuts.html): Global cut aggressiveness setting (1 = moderate cut generation).
    #     - [CliqueCuts](https://www.gurobi.com/documentation/9.1/refman/cliquecuts.html#parameter:CliqueCuts): Controls clique cut generation (1 = moderate cut generation). Overrides the Cuts parameter.
    #     - [ObjScale](https://www.gurobi.com/documentation/9.1/refman/objscale.html): when positive, divides the model objective by the specified value to avoid numerical issues that may result from very large or very small objective coefficients. The default value of 0 decides on the scaling automatically
    #     - [ScaleFlag](https://www.gurobi.com/documentation/9.1/refman/scaleflag.html): meaning of values described [here](https://support.gurobi.com/hc/en-us/community/posts/15330309039633-How-does-the-ScaleFlag-option-work)
    #     - [NoRelHeurWork](https://www.gurobi.com/documentation/9.5/refman/norelheurwork.html#parameter:NoRelHeurWork): similar to [NoRelHeurTime](https://www.gurobi.com/documentation/9.5/refman/norelheurtime.html) but provides deterministic results.
    # - [Solver Status](https://www.gurobi.com/documentation/current/refman/optimization_status_codes.htmlhttps://www.gurobi.com/documentation/current/refman/optimization_status_codes.html)
    # - Python Parameter examples: [Gurobi link](https://www.gurobi.com/documentation/current/refman/python_parameter_examples.html#PythonParameterExampleshttps://www.gurobi.com/documentation/current/refman/python_parameter_examples.html#PythonParameterExamples)
    # - Post on [*Why scaling and geometry is relevant*](https://www.gurobi.com/documentation/9.1/refman/why_scaling_and_geometry_i.html): 
    #     - "As we performed larger and larger rescalings, we continued to obtain the same optimal value, but there were clear signs that the solver struggled. However, once we pass a certain rescaling value, the solver is no longer able to solve the model and instead reports that it is Infeasible."
    # - Post on [*Model sometimes works, sometimes hangs indefinitely Ongoing*](https://support.gurobi.com/hc/en-us/community/posts/360074702452-Model-sometimes-works-sometimes-hangs-indefinitely)
    # - Post on [*MILP takes too much time*](https://support.gurobi.com/hc/en-us/community/posts/9761458650129-MILP-takes-too-much-time)
    
    print("Simulation started at %s. Generating data input." %datetime.datetime.now().time().strftime("%H:%M:%S"))

    # Energy Limits: Logistic curve parameters
    k_logi = 0.5 # Exponential
    L = 600e9 # Asymptote (GJ)
    t0 = 10 

    # ------------------------------------ Initialising model input dataframes ------------------------------------
    # Initialise the list which will save for this future ID the input data to the model  
    all_info = []

    for futureID in df_input.index:
    #for futureID in [0,1,2,3,4,5,6,7,8,9]:

        # Dictionary where all variables relevant for a future are stored. This will then be added to "conf_batch"
        dict_ID = {'T':t_ik,
                   }
        
        # Non-CDR-specific data:
        Non_CDR_data = pd.DataFrame(index=non_i, columns = ['P']) # As dictionary:  {{'CDRRequired':...},{'LandL':...}, etc}
        # Endogenous Parameters:
        Endo = pd.DataFrame(index=i, columns = il)
        Z_lo = dict.fromkeys(merged_il_list, 1)
        Z_up = dict.fromkeys(merged_il_list, 1)
        C_lo = dict.fromkeys(merged_il_list, 1)
        C_up = dict.fromkeys(merged_il_list, 1)
        beta = dict.fromkeys(merged_il_list, 1)

        # Resource Requirement Parameters:
        Resources = pd.DataFrame(index=i, columns = j)
        Land_r = dict.fromkeys(merged_list, 1)
        Energy_r = dict.fromkeys(merged_list, 1)
        OPEX_ik = dict.fromkeys(merged_list, 1)
        Xmax_input = dict.fromkeys(merged_list, 500e6) # Defautlt tCO2/year
        Initial_YRef = dict.fromkeys(i, 0)
        Ymax_input = dict.fromkeys(i, 1)

        # ------------------------------------ Extracting data  ------------------------------------
        # For each row (=future_ID), save value of each parameter (=column) in the LHS sample table 
        for col in df_input.columns:
            name = col.split('_')
            # This is for checking whether the current parameter (column) is a CDR-indexed parameter or not:
            if name[0] in non_i:
                Non_CDR_data.loc[name[0]] = int(df_input.loc[futureID,col])
            # This is for checking whether the current parameter (column) is an endogenous CDR-indexed parameter or not
            elif name[0] in il: 
                for CDR in i:
                    if (name[1] == CDR):
                        Endo.loc[CDR,name[0]] = df_input.loc[futureID,col] # --> save them in Endo dataframe 
            # This is for the resource CDR-specific parameters --> save them in Resources dataframe
            else:
                for res in j:
                    for CDR in i:
                        if (name[0] == res) & (name[1] == CDR):
                            Resources.loc[CDR,res] = df_input.loc[futureID,col]

        # ------------------------------------ Computing Relevant parameters ------------------------------------
        Initial_YRef.update({CDR:int(Endo.loc[CDR,'Yref']) for CDR in i})
        Ymax_input.update({CDR:int(Endo.loc[CDR,'Ymax']) for CDR in i})
        Endo["ER"] = Endo["ER"] # Required to divide by 100 as input data is in units of [%]
        Endo['b'] = -np.log(1-Endo.ER.astype(np.float64))/np.log(2) 
            # Needed this extra "as type" due to this: https://stackoverflow.com/questions/47208473/attributeerror-numpy-float64-object-has-no-attribute-log10
        # Compute Initial Investment Costs for DOCCS based on future's Experience Parameter 
        Endo.loc['DOCCS','SCref'] = int(unit_costs_b(Endo.loc['DOCCS','Yref'], Endo.loc['DOCCS','b'], Endo.loc['DACCS','Yref'], Endo.loc['DACCS','SCref']))

        # Energy Limit 
        av = int((Non_CDR_data.loc['EnergyL','P']-first_year)/delK) # Time index at which energy will start being available  
        ELimit = dict.fromkeys(list(range(k1,T)),0) # Initialise dictionarry for storing 
        #func = [1e9*np.exp(_*0.5) for _ in np.linspace(0,(T-av-1),T-av)]# Exponential function
        func = [L/(1+np.exp(-k_logi*(_-t0))) for _ in np.linspace(0,(T-av-1),T-av)] # Logistic function
        ELimit.update({n:int(func[n-av]) for n in range(av,T)})

        for CDR in i:
            # Resource Requirements
            Land_r.update({(CDR,n):Resources.loc[CDR,'Land'] for n in range(k1,T)})
            Energy_r.update({(CDR,n):Resources.loc[CDR,'Energy'] for n in range(k1,T)})
            OPEX_ik.update({(CDR,n):int(Resources.loc[CDR,'OPEX']) for n in range(k1,T)})
            # Annual Limit to capacity expansion
            Xmax_input.update({(CDR,n):int(Resources.loc[CDR,'Xmax']) for n in range(k1,T)}) 

            # ------------- Piecewise Linear Parameters -------------
            # Per CDR option, compute the maximum experience that could be achieved, i.e., the experience that would result if the maximum capacity X^max was deployed at all active time periods
            ZUp_max = (Endo.loc[CDR,'Yref']+Xmax_input[CDR,T-1]*delK*(active_k[CDR]-1))
            # For l=1
            Z_lo.update({(CDR,1):int(Endo.loc[CDR,'Yref'])}) # Lowest point corresponds to already installed capacity
            C_lo.update({(CDR,1):0}) # Initial cumulative costs are 0 (as capacity is already installed)

            # For l=2,3,4
            for n in [2,3,4]:
                Z_lo.update({(CDR,n):int(Z_lo[CDR,n-1]+Xmax_input[CDR,T-1]*delK)}) # Segments are defined in terms of maximum deployed annual capacity
            C_lo.update({(CDR,n):int((Endo.loc[CDR,'SCref']/(1-Endo.loc[CDR,'b']))*((((Z_lo[CDR,n])**(1-Endo.loc[CDR,'b']))/(Endo.loc[CDR,'Yref']**(-Endo.loc[CDR,'b'])))-Endo.loc[CDR,'Yref'])) for n in [2,3,4]})

            Z_up.update({(CDR,n):int(Z_lo[CDR,n+1]-1) for n in [1,2,3]}) # Upper point of segment l is equal to lower point of segment l+1 
            Z_up.update({(CDR,4):int(ZUp_max)})

            C_up.update({(CDR,n):int((Endo.loc[CDR,'SCref']/(1-Endo.loc[CDR,'b']))*((((Z_up[CDR,n])**(1-Endo.loc[CDR,'b']))/(Endo.loc[CDR,'Yref']**(-Endo.loc[CDR,'b'])))-Endo.loc[CDR,'Yref'])) for n in l})

            beta.update({(CDR,n):int((C_up[CDR,n]-C_lo[CDR,n])/(Z_up[CDR,n]-Z_lo[CDR,n])) for n in l}) # Computing the segments slope     
        #plotting_points(i,Endo,Z_lo,Z_up,C_lo,C_up,wanted,Xmax_input)

        # Saving all of the data to a dictionary
        dict_ID['G'] = {None: int(Non_CDR_data.loc['CDRRequired','P'])}
        dict_ID['Q'] = dict.fromkeys(list(range(k1,T)), int(Non_CDR_data.loc['GeoSt','P']))

        dict_ID['yLo'] = Z_lo 
        dict_ID['yUp'] = Z_up
        dict_ID['CLo'] = C_lo
        dict_ID['CUp'] = C_up
        dict_ID['Beta'] = beta

        dict_ID['Ymax'] = Ymax_input
        dict_ID['Yini'] = Initial_YRef

        dict_ID['LandL'] = dict.fromkeys(list(range(k1,T)), int(Non_CDR_data.loc['LandL','P'])) # Limit on land in period k (ha)
        dict_ID['EnergyL'] = ELimit # Calculated above: 0 until "EnergyL" year, then grows following logistic function]
        dict_ID['LandR'] = Land_r
        dict_ID['EnergyR'] = Energy_r
        dict_ID['OPEX'] = OPEX_ik
        dict_ID['Xmax'] = Xmax_input
        #dict_ID['T'] = t_ik # Already defined above, first line of the 'for' loop

        all_info += [[futureID, dict_ID]]
    
    print("Done with preparing input data - let's start solving for futures at:", datetime.datetime.now().time().strftime("%H:%M:%S"))

    # ======================================================================================================================
    #for futureID in df_input.index: # for all the future IDs in the table input
    #for futureID in a:
    '''def solving_future(futureID, results_y,results_x,results_b,results_z,results_Costs,results_C,results_E,results_L,results_obj,
                    solved_id, solved_type, solved, not_solved, solved_binary):'''

    #futureID,non_i,merged_il_list,merged_list,i,j,il,df_input,unit_costs_b,first_year,delK,k_logi,t0,T,plotting_points,wanted,k1,t_ik,Tee,active_k
    # conf_batch = [[futureID, big_conf],]
    #big_conf = {futre_ID:}

    def parallel_process(conf_batch, n_jobs=16, front_num=3):
        if front_num > 0:
            front = [solving_future(conf) for conf in conf_batch[:front_num]]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool: # https://stackoverflow.com/questions/806499/threading-vs-parallelism-how-do-they-differ
            futures = [pool.submit(solving_future,conf)
                for conf in conf_batch[front_num:]
            ]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
        out = []
        #Get the results from the futures. 
        for index, future in tqdm(enumerate(futures)):
            try:
                out.append(future.result())
            except Exception as e:
                out.append(e)
        pool.shutdown()
        return front + out 

    start_time = time.perf_counter() # Initialise time

    results_parallel = parallel_process(all_info, n_jobs=20, front_num=3)
    # given that 'solving_future' returns 'results = [futureID, dict_ID]', aaa = [results, results] so that 'aaa[0] = results = [0, {'y': [{('BECCS', 0): 1820000.0000000002, ('BECCS', 1): ...]'
    
    solved_binary = {item[0] : item[2] for item in results_parallel}

    print("Optimisation finished at %s. Elapsed time: %.0f seconds (%.0f minutes). Out of %.0f futures evaluated, %.0f have been solved. Starting data saving." 
          %(datetime.datetime.now().time().strftime("%H:%M:%S"), (time.perf_counter() - start_time), (time.perf_counter() - start_time)/60, len(results_parallel), sum(1 for v in solved_binary.values() if v == 1)))
    
    # ======================================================================================================================

    pb = pd.read_excel("PortfolioFiles/Portfolio_Files_Latest/extra_resources.xlsx", index_col = "CDR_option")

    # Dataframes to save results
    resources = pd.DataFrame()
    results = pd.DataFrame()
    total_future = pd.DataFrame()
    max_capacity = pd.DataFrame()  # Maximum capacity 
    max_experience = pd.DataFrame() # Maximum experience
    max_resources = pd.DataFrame() 
    removed_per_CDR = pd.DataFrame()
    
    results_2050 = pd.DataFrame()
    aggregated_2050 = pd.DataFrame()

    # Dataframes: (Max capacity in indicated time period) - (Initial Capacity Installed) 
    max_2050 = pd.DataFrame(index = i)
    max_2075 = pd.DataFrame(index = i)
    max_2100 = pd.DataFrame(index = i)

    for item in range(0,len(results_parallel)): # For the items in the order of the resutls
        futureID = results_parallel[item][0] # Future ID is the first item in a list
        dict_results = results_parallel[item][1]
        
        resources_in_future = pd.DataFrame(index=[(futureID, k[m]) for m in range(k1, len(k))])
        resources_in_future['Land Required'] = dict_results['ld'][0].values() #results_L[futureID].values()
        resources_in_future['Energy Required'] = dict_results['e'][0].values() #results_E[futureID].values()
        resources = pd.concat([resources, resources_in_future])
        
        results_in_future = pd.DataFrame(index=[(futureID, i[n], k[m]) for n in range(0, len(i)) for m in range(k1, len(k))])
        results_in_future['Cumulative capacity'] = dict_results['y'][0].values() #results_y[futureID].values()
        results_in_future['Annual Capacity'] = dict_results['x'][0].values() #results_x[futureID].values()
        results_in_future['Removals per period'] = dict_results['b'][0].values() #results_b[futureID].values()
        results_in_future['Experience'] = dict_results['z'][0].values() #results_z[futureID].values()
        results_in_future['Costs per period (undiscounted)'] = dict_results['costs'][0].values() #results_Costs[futureID].values()
        results_in_future['End. Cumulative Costs'] = dict_results['c'][0].values() # results_C[futureID].values()

        total_future.loc[futureID,'Removals'] = results_in_future['Removals per period'].sum(axis=0)
        total_future.loc[futureID,'Costs'] = dict_results['OBJ'][0]

        # Saving the removals per CDR option 
        removed_in_future = pd.DataFrame(columns= [i[n] for n in range(0, len(i))])
        for CDR in i:
            removed = 0
            for m in range(k1, len(k)):
                #removed += results_in_future.loc[(futureID,CDR,m),'Removals per period']
                removed += dict_results['b'][0][CDR,m]
            removed_in_future.loc[futureID,CDR] = removed
        removed_per_CDR = pd.concat([removed_per_CDR, removed_in_future])

        results = pd.concat([results, results_in_future])
        
        results_in_future.index = pd.MultiIndex.from_tuples(results_in_future.index,names=['future_id','CDR_opt','k'])
        max_capacity = pd.concat([max_capacity, pd.DataFrame(results_in_future.groupby(level=1).max()['Cumulative capacity']).rename(columns={"Cumulative capacity":futureID})], axis=1)
        max_experience = pd.concat([max_experience, pd.DataFrame(results_in_future.groupby(level=1).max()['Experience']).rename(columns={"Experience":futureID})], axis=1)
        
        max_resources.loc[futureID,'Land'] = resources_in_future['Land Required'].max()
        max_resources.loc[futureID,'Energy'] = resources_in_future['Energy Required'].max()
        
        results_in_2050 = results_in_future[results_in_future.index.get_level_values('k') <= 6]
        results_2050 = pd.concat([results_2050, results_in_2050])
        
        aggregated_2050.loc[futureID,'Removals'] = results_in_2050['Removals per period'].sum(axis=0)
        aggregated_2050.loc[futureID,'Undiscounted Costs'] = results_in_2050['Costs per period (undiscounted)'].sum(axis=0)

        a = results_in_future.droplevel('future_id')
        max_future_2050 = pd.DataFrame(index = i)
        max_future_2075 = pd.DataFrame(index = i)
        max_future_2100 = pd.DataFrame(index = i)

        for CDR in i:
            # Saving data for 3 different time periods:
            maxxxx = a.loc[(CDR,0),'Cumulative capacity'] - Initial_YRef[CDR]       
            for _ in range(1,7): # i.e., from 2025 to 2050
                if (a.loc[(CDR,_),'Cumulative capacity'] - Initial_YRef[CDR]) >= maxxxx:
                    maxxxx = a.loc[(CDR,_),'Cumulative capacity'] - Initial_YRef[CDR]
            max_future_2050.loc[CDR,futureID] = round(maxxxx,0)
            
            for _ in range(7,12): # i.e., from 2055 to 2075
                if (a.loc[(CDR,_),'Cumulative capacity'] - Initial_YRef[CDR]) >= maxxxx:
                    maxxxx = a.loc[(CDR,_),'Cumulative capacity'] - Initial_YRef[CDR]
            max_future_2075.loc[CDR,futureID] = round(maxxxx,0)
            
            for _ in range(12,T): # i.e., from 2080 to 2100
                if (a.loc[(CDR,_),'Cumulative capacity'] - Initial_YRef[CDR]) >= maxxxx:
                    maxxxx = a.loc[(CDR,_),'Cumulative capacity'] - Initial_YRef[CDR]
            max_future_2100.loc[CDR,futureID] = round(maxxxx,0)
        
        max_2050 = pd.concat([max_2050, max_future_2050], axis = 1)
        max_2075 = pd.concat([max_2075, max_future_2075], axis = 1)        
        max_2100 = pd.concat([max_2100, max_future_2100], axis = 1)        
        
        max_resources.loc[futureID,'Water'] = (pb['Water']*(a['Removals per period'].groupby(level='CDR_opt').sum())).sum(axis=0)
        max_resources.loc[futureID,'Nitrogen'] = (pb['N']*(a['Removals per period'].groupby(level='CDR_opt').sum())).sum(axis=0)
        max_resources.loc[futureID,'Phosphorous'] = (pb['P']*(a['Removals per period'].groupby(level='CDR_opt').sum())).sum(axis=0)
        
    max_capacity = max_capacity.T
    max_experience = max_experience.T

    aggregated_results = df_LHS.T
    aggregated_results['Solved'] = pd.Series(solved_binary)

    total_future.index.names = ['future_id']
    max_resources.index.names = ['future_id']
    max_capacity.index.names = ['future_id']

    # Formatting to save results into csv file:
    max_years = pd.DataFrame()
    time_interest = max_2050.copy()
    for year in [2050, 2075, 2100]:
        if year == 2075:
            time_interest = max_2075.copy()
        if year == 2100:
            time_interest = max_2100.copy()
        time_interest = time_interest.T
        time_interest.columns = [str(col) + '_%d' %year for col in time_interest.columns]
        max_years = pd.concat([max_years,time_interest], axis=1) 
    max_years.index.names = ['future_id']
    
    removed_per_CDR.columns = [str(col) + '_removed' for col in removed_per_CDR.columns]

    # Add all relevant dataframes:
    aggregated_results = pd.concat([aggregated_results,total_future,max_capacity,max_resources,max_years,removed_per_CDR], axis=1)
    aggregated_results.index.names = ['future_id']

    aggregated_results.to_csv(r'Input_Output_Files/aggregated_results_%s.csv' %date.today().strftime("%d.%m.%Y"), index=True)
    
    all_metrics_names = dict(zip(aggregated_results.columns.to_list(),['DACCS Energy Requirements','DOCCS Energy Requirements','EW Energy Requirements', 'OA Energy Requirements', 'Year at which energy becomes available for CDR',
    'A/R Land Requirements','BECCS Land Requirements','Land Limit','A/R Experience Parameter','BC Experience Parameter','BECCS Experience Parameter','DACCS Experience Parameter',
    'DOCCS Experience Parameter','EW Experience Parameter','OA Experience Parameter','SCS Experience Parameter','Removals Required (2020 - 2100)','A/R Maximum Potential', 'BC Maximum Potential', 'BECCS Maximum Potential',
    'DACCS Maximum Potential', 'DOCCS Maximum Potential', 'EW Maximum Potential', 'OA Maximum Potential', 'SCS Maximum Potential', 'Annual Limit to Geological Injection','Solved Futures', 'Tons Removed', 'Discounted Cumulative Costs',
    'Maximum Capacity of AR','Maximum Capacity of BC','Maximum Capacity of BECCS','Maximum Capacity of DACCS','Maximum Capacity of DOCCS', 'Maximum Capacity of EW', 'Maximum Capacity of OA', 'Maximum Capacity of SCS',
    'Land used for CDR', 'Energy used for CDR','Water used for CDR', 'Nitrogen used for CDR', 'Phosphorous used for CDR','Maximum Capacity BECCS by 2050','Maximum Capacity A/R by 2050','Maximum Capacity SCS by 2050','Maximum Capacity BC by 2050','Maximum Capacity DACCS by 2050',
    'Maximum Capacity EW by 2050','Maximum Capacity OA by 2050','Maximum Capacity DOCCS by 2050','Maximum Capacity BECCS by 2075','Maximum Capacity A/R by 2075','Maximum Capacity SCS by 2075','Maximum Capacity BC by 2075','Maximum Capacity DACCS by 2075',
    'Maximum Capacity EW by 2075','Maximum Capacity OA by 2075','Maximum Capacity DOCCS by 2075','Maximum Capacity BECCS by 2100','Maximum Capacity A/R by 2100','Maximum Capacity SCS by 2100','Maximum Capacity BC by 2100','Maximum Capacity DACCS by 2100',
    'Maximum Capacity EW by 2100','Maximum Capacity OA by 2100','Maximum Capacity DOCCS by 2100']))

    all_metrics_units = dict(zip(aggregated_results.columns.to_list(),['GJ/tCO2','GJ/tCO2','GJ/tCO2','GJ/tCO2', 'year', 'ha/tCO2','ha/tCO2','ha',
    '%','%','%','%','%','%','%','%','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2/yr','binary', 'tCO2', 'USD','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2','tCO2',
    'ha', 'GJ','km3', 'Mt N', 'Mt P','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr',
    'tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr','tCO2/yr']))

    all_metrics_ranges = {key: [] for key in aggregated_results.columns.to_list()}
    for metric in aggregated_results.columns:
        all_metrics_ranges[metric].append([aggregated_results[metric].min(),aggregated_results[metric].max()])

    print('Data has been saved at:', datetime.datetime.now().time().strftime("%H:%M:%S"))

    return aggregated_results

if __name__ == '__main__':
    main()