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
from pathlib import Path


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
    - input_file_path: Path of input file to import
    - n_sample: int, number of samples
    - dict_constant: Dictionary of values that remain constant in the form of {'Ymax'=1}
    - sampling_method: "Step" or "Size"
    - future_zero: True or False
    - output_file: 
'''

class sampling_futures:
    def __init__(self, input_file_path, n_sample, dict_constant, sampling_method, future_zero, output_file, output_type):
        self.input_file = input_file_path
        self.output_file = output_file
        self.output_type = output_type

        self.future_zero = future_zero # Determines whether a "all min" future wants to be added at the beginning
        
        self.n_lhs = n_sample # This refers to the number of times the LHS method will be sampled
        if future_zero == True: 
            self.n_sample = n_sample + 1 # self.n_sample refers to the number of futures in the output file
        else: 
            self.n_sample = n_sample # self.n_sample refers to the number of futures 

        self.dict_constant = dict_constant # Dictionary of values that remain constant
        self.sampling_method = sampling_method[0] # can take the values of 'step' or 'size'
        
        # ===================================== Importing and Formating dataframe =====================================
        df = pd.read_excel(input_file_path, index_col = "Parameter")
        self.units = dict(zip(df.index, df.Unit)) # Save all units in a dictionary for later use
        # Separate all constant parameters
        constant = df[df.minimum == df.maximum]
        constant = constant[['minimum']].T
        self.constant = pd.concat([constant]*(self.n_sample),axis=0,ignore_index=True)

        # Get a data frame columns to be sampled columns
        df = df[df.minimum != df.maximum]

        for row in df.index:
            if df.loc[row, 'Resource'] in sampling_method[1]['not_sampled']:
                pass
            else:   
                if self.sampling_method == 'size':
                    df.loc[row,'Sampling_step'] = sampling_method[1][self.sampling_method][df.loc[row, 'Resource']]
                    df.loc[row,'N_samples'] = 1 + (float(df.loc[row,'maximum'])-float(df.loc[row,'minimum']))/df.loc[row,'Sampling_step']
                
                if self.sampling_method == 'step':
                    if df.loc[row, 'Resource'] in sampling_method[1]['exceptions']:
                        df.loc[row,'Sampling_step'] = sampling_method[1]['exceptions'][df.loc[row, 'Resource']]
                        df.loc[row,'N_samples'] = 1 + (float(df.loc[row,'maximum'])-float(df.loc[row,'minimum']))/df.loc[row,'Sampling_step']
                        
                    else:
                        df.loc[row,'N_samples'] = sampling_method[1][self.sampling_method][df.loc[row, 'Resource']]
                        df.loc[row,'Sampling_step'] = (float(df.loc[row,'maximum'])-float(df.loc[row,'minimum']))/(df.loc[row,'N_samples']-1)
        
        self.sampled_parameters = df.index.to_list()
        self.df_Sampling_step = df[['Sampling_step']]

        self.df = df
    
    def sample(self):
        if self.sampling_method == 'step':
            # Modify dataframe by Sampling step
            df = self.df[['minimum','maximum']]
            df = df.div(self.df_Sampling_step.Sampling_step, axis=0)

            dict_ranges = df.T.to_dict('records')
            dict_ranges = dict_ranges[0]

            lbounds = df["minimum"].values.tolist()
            lbounds = [int(x) for x in lbounds]

            ubounds  = df["maximum"].values.tolist()
            ubounds = [int(x) for x in ubounds]

            sampler = qmc.LatinHypercube(d=len(lbounds))
            sample = sampler.integers(l_bounds=lbounds, u_bounds=ubounds, n=self.n_lhs, endpoint=True)

            df_LHS = pd.DataFrame.from_records(sample)    
            df_LHS = df_LHS.set_axis(dict_ranges, axis=1)
            
            # If future_zero == True, add the min values for "future ID_0"
            if self.future_zero == True: df_LHS = pd.concat([df[["minimum"]].T,df_LHS], axis = 0, ignore_index=True) # Add the min values for "future ID_0"
            
            df_LHS.index.names = ['future_id']

            # Return to original units
            df_LHS = df_LHS.T.mul(self.df_Sampling_step.Sampling_step, axis=0).astype(float).round(3)

            # Creating model input table
            df_input = pd.concat([df_LHS.T, self.constant], axis=1)
            df_input.index.names = ['future_id']
            
            if self.output_type == 'xlsx': df_input.to_excel(self.output_file, index=True)
            if self.output_type == 'csv': df_input.to_csv(self.output_file, index=True)

            return df_input, self.units, self.sampled_parameters
        
        elif self.sampling_method == 'size':
            df_Number_samples = self.df[['N_samples']].rename(columns={'N_samples':"maximum"})
            df_Number_samples[["minimum"]] = 1
            df = self.df[['minimum','maximum']]

            dict_ranges = df.T.to_dict('records')[0]

            lbounds = df_Number_samples["minimum"].values.tolist()
            lbounds = [int(x) for x in lbounds]

            ubounds  = df_Number_samples["maximum"].values.tolist()
            ubounds = [int(x) for x in ubounds]

            sampler = qmc.LatinHypercube(d=len(lbounds))
            sample = sampler.integers(l_bounds=lbounds, u_bounds=ubounds, n=self.n_lhs, endpoint=True)

            df_LHS = pd.DataFrame.from_records(sample)    
            df_LHS = df_LHS.set_axis(dict_ranges, axis=1)

            # Return to original units
            #df_LHS = df_LHS.T.mul(df_Sampling_step.Sampling_step, axis=0)
            for futureID in range(0,self.n_lhs):
                for parameter in df_LHS.columns:
                    if df_LHS.loc[futureID,parameter] == 1:
                        df_LHS.loc[futureID,parameter] = self.df.loc[parameter,"minimum"]
                    else:
                        df_LHS.loc[futureID,parameter] = round(self.df.loc[parameter,"minimum"] + (df_LHS.loc[futureID,parameter]-1)*self.df_Sampling_step.loc[parameter,"Sampling_step"], 3)
            
            # If future_zero == True, add the min values for "future ID_0"
            if self.future_zero == True: df_LHS = pd.concat([df[["minimum"]].T,df_LHS], axis = 0, ignore_index=True) 
            df_LHS.index.names = ['future_id']

            # Creating model input table
            df_input = pd.concat([df_LHS,self.constant], axis=1)
            df_input.index.names = ['future_id']
            df_LHS = df_LHS.T
            
            if self.output_type == 'xlsx': df_input.to_excel(self.output_file, index=True)
            if self.output_type == 'csv': df_input.to_csv(self.output_file, index=True)
            
            return df_input, self.units, self.sampled_parameters
        
        else:
            print()
            raise ValueError("%s - The sampling_method arguments accepts either 'step' or 'size'." %self.sampling_method)
