# CDR_Portfolios

## Installation
We use conda for package management
```
conda env create --file environment.yml
conda activate cdr_portfolio_model
```

Additionally to these requirememts, to solve the mixed-integer linear programming problem, this code uses Gurobi's solver, available through an academic license at:  

## Replicating Results

Replicating the results involves the following steps:
    1. Importing uncertain ranges and creating a future sample (<creating_futures_sample.py>). This will be the input file to step 2. Alternatively, one can directly use an existing input file and skip Step 1.
        - This file reads data from "config.yaml"  
    2. Solving futures from an input file (<Model_Solving_Futures_Class.py>). This will read an input (.csv or .xlsl) file and compute results for each future using Gurobi solver.  
        - This file reads data from "key_parameters.yaml" 
        - It is NOT recommended to run <Model_Solving_Futures_Class.py> on a Jupyter notebook, but rather directly from the Terminal  
    3. Analysis and visualisation of results: 
        - This file reads data from "metric_names.yaml"
