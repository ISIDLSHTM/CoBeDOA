# CoBe Dose Optimsation Approach 
Supplementary data and code for correlated beta dose optimisation investigation.


## Order of operations
0. Choose how many simulation you would like to run by updating num_sims in config.py. Default is 100.
1. Install all requirements (see requires.txt)
2. Run pickling_for_environments\pickling.py to generate the data for the scenarios. 
3. Run Running\Objective_1\_Conduct_objective.py 
4. Run Running\Objective_2\_Conduct_objective.py 
5. Run Running\Objective_3\_Conduct_objective.py 
6. Run Running\Objective_4\_Conduct_objective.py 

All of these steps will require many hours of compute time. For steps 3-6 this can be reduced by reducing num_sims to lower value, though this will mean that your results may be more effected by stochastic effects and you will have reduced precision. 

Data that were generated for use in the paper are in a seperate repo at github.com/ISIDLSHTM/CoBeDOA_Data
