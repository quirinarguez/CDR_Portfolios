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
from IPython.display import display
import yaml



from tqdm import tqdm # Source: https://danshiebler.com/2016-09-14-parallel-progress-bar/ --> Thanks!
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys # Python "sys" documentation: https://docs.python.org/3/library/sys.html

# To reset the default parameters of matplotlib:
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

'''
# Python Class: Syntax and Examples [Python Tutorial]: https://mimo.org/glossary/python/class

The class "sampling_futures" takes as inputs:
    - aaa
    - 
'''

class sampling_futures:
    def __init__(self, key_parameters, input_file_path, n_sample, dict_constant, sampling_method, future_zero, output_file):
        self.input_file = input_file_path
        self.output_file = output_file
        self.future_zero = future_zero # Determines whether a "all min" future wants to be added at the beginning
        
        self.n_lhs = n_sample # This refers to the number of times the LHS methos will be sampled
        if future_zero == True: 
            self.n_sample = n_sample + 1 # self.n_sample refers to the number of futures 
        else: 
            self.n_sample = n_sample # self.n_sample refers to the number of futures 
        
        self.i = key_parameters['i']
        '''MISSING PARAMETERS...
        k1 = 0 # Value of first time period
        delK = 5 # Time step of optimisation (i.e., length of each period)
        N = 4 # Number of segments for piecewise linear approximation
        first_year = 2020 # Year corresponding to first time period
        last_year = 2100 # Year corresponding to last time period
        T = int(1 + (last_year-first_year)/delK) # Number of time periods evaluated, equivalent to 17 if delK = 5 (i.e., 80 years)
        k = list(range(k1,T))'''

        self.dict_constant = dict_constant
        self.sampling_method = sampling_method # can take the values of 'step' or 'size'
    
        # ===================================== Importing and Formating dataframe =====================================
        df = pd.read_excel(self.input_file, index_col = "Parameter")
        self.units = dict(zip(df.index, df.Unit)) # Save all units in a dictionary for later use
        # Separate all constant parameters
        constant = df[df.minimum == df.maximum]
        constant = constant[['minimum']].T
        self.constant = pd.concat([constant]*(self.n_sample),axis=0,ignore_index=True)

        # Get a data frame with selected columns
        df = df[df.minimum != df.maximum]
        self.df_Sampling_step = df[['Sampling_step']]

        self.df = df
    
    def sample(self):
        if self.sampling_method == 'step':
            # Modify dataframe by Sampling step
            self.df = self.df[['minimum','maximum']]
            self.df = self.df.div(self.df_Sampling_step, axis=0)

            dict_ranges = self.df.T.to_dict('records')
            dict_ranges = dict_ranges[0]

            lbounds = self.df["minimum"].values.tolist()
            lbounds = [int(x) for x in lbounds]

            ubounds  = self.df["maximum"].values.tolist()
            ubounds = [int(x) for x in ubounds]

            sampler = qmc.LatinHypercube(d=len(lbounds))
            sample = sampler.integers(l_bounds=lbounds, u_bounds=ubounds, n=self.n_lhs, endpoint=True)

            df_LHS = pd.DataFrame.from_records(sample)    
            df_LHS = df_LHS.set_axis(dict_ranges, axis=1)
            
            # If future_zero == True, add the min values for "future ID_0"
            if self.future_zero == True: df_LHS = pd.concat([self.df[["minimum"]].T,df_LHS], axis = 0, ignore_index=True) # Add the min values for "future ID_0"
            
            df_LHS.index.names = ['future_id']

            # Return to original units
            df_LHS = df_LHS.T.mul(self.df_Sampling_step.Sampling_step, axis=0)

            # Creating model input table
            df_input = pd.concat([df_LHS.T, self.constant], axis=1)
            df_input.index.names = ['future_id']
            display(df_input.head(5))

            return df_input, units, dict_ranges
        
        elif self.sampling_method == 'step':
            df_Number_samples = self.df[['N_samples']].rename(columns={'N_samples':"maximum"})
            df_Number_samples[["minimum"]] = 1
            self.df = self.df[['minimum','maximum']]

            dict_ranges = self.df.T.to_dict('records')[0]

            lbounds = df_Number_samples["minimum"].values.tolist()
            lbounds = [int(x) for x in lbounds]

            ubounds  = df_Number_samples["maximum"].values.tolist()
            ubounds = [int(x) for x in ubounds]

            sampler = qmc.LatinHypercube(d=len(lbounds))
            sample = sampler.integers(l_bounds=lbounds, u_bounds=ubounds, n=self.n_lhs, endpoint=True)

            df_LHS = pd.DataFrame.from_records(sample)    
            df_LHS = df_LHS.set_axis(dict_ranges, axis=1)

            # If future_zero == True, add the min values for "future ID_0"
            if self.future_zero == True: df_LHS = pd.concat([self.df[["minimum"]].T,df_LHS], axis = 0, ignore_index=True) 
            df_LHS.index.names = ['future_id']

            # Return to original units
            #df_LHS = df_LHS.T.mul(df_Sampling_step.Sampling_step, axis=0)
            for futureID in range(1,self.n_lhs+1):
                for parameter in df_LHS.columns:
                    if df_LHS.loc[futureID,parameter] == 1:
                        df_LHS.loc[futureID,parameter] = self.df.loc[parameter,"minimum"]
                    else:
                        df_LHS.loc[futureID,parameter] = round(self.df.loc[parameter,"minimum"] + (df_LHS.loc[futureID,parameter]-1)*self.df_Sampling_step.loc[parameter,"Sampling_step"], 3)

            # Creating model input table
            df_input = pd.concat([df_LHS,self.constant], axis=1)
            df_input.index.names = ['future_id']
            df_LHS = df_LHS.T

            display(df_input.head(5))

            return df_input, units, dict_ranges
        
        else:
            print()
            raise ValueError("%s - The sampling_method argumetns accepts either 'step' or 'size'." %self.sampling_method)


    '''Uncertainties = dict(zip(df_LHS.index.to_list(),['DACCS Energy Requirements','DOCCS Energy Requirements','EW Energy Requirements', 'OA Energy Requirements', 'Year at which energy becomes available for CDR',
    'A/R Land Requirements','BECCS Land Requirements','Land Limit','A/R Experience Parameter','BC Experience Parameter','BECCS Experience Parameter','DACCS Experience Parameter',
    'DOCCS Experience Parameter','EW Experience Parameter','OA Experience Parameter','SCS Experience Parameter','Removals Required (2020 - 2100)','A/R Maximum Potential', 'BC Maximum Potential', 'BECCS Maximum Potential',
    'DACCS Maximum Potential', 'DOCCS Maximum Potential', 'EW Maximum Potential', 'OA Maximum Potential', 'SCS Maximum Potential', 'Annual Limit to Geological Injection']))'''