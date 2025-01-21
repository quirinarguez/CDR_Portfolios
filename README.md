# CDR_Portfolios

## Installation
We use conda for package management
```
conda env create --file environment.yml
conda activate cdr_portfolio_model
```

Additionally to these requirements, to solve the mixed-integer linear programming problem, this code uses Gurobi's solver, available through an academic license at: https://www.gurobi.com/academia/academic-program-and-licenses/  

## Replicating Results

Replicating the results involves the following steps:

    1. Importing uncertain ranges and creating a future sample (<creating_futures_sample.py>). This will be the input file to step 2. Alternatively, one can directly use an existing input file from 'PortfolioFiles/sample_of_futures' and skip Step 1.
        - This file reads data from "config.yaml" and can be run directly through the Terminal using the <main> function at the end of the file. 
    
    2. Solving futures from an input file (<Model_Solving_Futures_Class.py>). This will read an input (.csv or .xlsl) file and compute results for each future using Gurobi solver.  
        - This file reads data from "key_parameters.yaml" and from "config.yaml"
        - This file outputs data to a csv file to the 'PortfolioFiles/results_from_modelling/' folder
        - If the relevant lines are uncommented, this file also outputs data to 'metric_names.yaml'  
        - IMPORTANT: It is NOT recommended to run <Model_Solving_Futures_Class.py> on a Jupyter notebook, but rather directly from the Terminal. To do this, one can determine the data required to run the code either in "config.yaml" or directly in the <main> function at the end of the file.
    
    3. Analysis and visualisation of results (Analysis_CDR_Portfolios.ipynb): 
        - This file reads data from "metric_names.yaml" and from the resultant .csv file from Step 2. 
