from utility_functions import maximise_efficacy
from models import CoBe_Model, Null_Model, Latent_Quadratic_Model
from experiment_class import Experiment
from selection_methods import thompson_sampling, softmax, uniform
from final_selection_methods import maximise_median, naive_best
from Functions_for_modelling.kernels import squared_exponential_kernel, uncorrelated_kernel

from environments import Environment_1_101_1, Environment_1_101_2, Environment_1_101_3, Environment_1_101_4, \
    Environment_1_101_5, Environment_1_101_6, Environment_1_101_7
from sql_functions_and_classes.sql_classes import *
from sql_functions_and_classes import sql_functions

import sqlite3
import numpy as np
from tqdm import tqdm
from config import CFG
import useful_functions

number_of_simulated_trials = CFG.num_sims
data_base_address = '__Objective_1_Database.db'

cohort_size = 6
number_of_cohorts_per_trial = 50
_d = np.array([0, .2, .4, .6, .8, 1]).reshape(-1, 1)

env_names = ['Environment_1', 'Environment_2', 'Environment_3',
             'Environment_4', 'Environment_5', 'Environment_6',
             'Environment_7']
exp_envs = [Environment_1_101_1, Environment_1_101_2, Environment_1_101_3,
            Environment_1_101_4, Environment_1_101_5, Environment_1_101_6,
            Environment_1_101_7]

approach_name = 'CoBe'
for environment_name, Exp_Environment in zip(env_names, exp_envs):
    conn = sqlite3.connect(data_base_address)
    c = conn.cursor()
    c.execute("BEGIN")
    most_recent = sql_functions.find_most_recent_experiment(conn)
    Experiment_ID = most_recent

    for repeat in tqdm(range(number_of_simulated_trials)):

        # Set Up Experiment ID and Random Seed
        Experiment_ID = Experiment_ID + 1
        np.random.seed(Experiment_ID)

        # Set up Models, Utility Function, and Selection Methods
        Efficacy_Model = CoBe_Model(Exp_Environment.query_doses, kernel=squared_exponential_kernel(.2))
        Toxicity_Model = Null_Model(Exp_Environment.query_doses)
        Utility_Function = maximise_efficacy()
        Selection_Method = thompson_sampling()
        Final_Selection_Method = maximise_median()

        # Initialise experiment class
        Exp = Experiment(Exp_Environment, Efficacy_Model, Toxicity_Model,
                         Utility_Function, Selection_Method, Final_Selection_Method,
                         cohort_size)

        # Iterate for number of trial blocks
        Exp.full_loop(number_of_cohorts_per_trial)

        # Insert Experiment into SQL Queue
        exp_record = Experiment_Record(Experiment_ID, approach_name, environment_name)
        sql_functions.insert_experiment(exp_record, c)

        # Insert a row into the SQL queue for each individuals
        indi_list = sql_functions.experiment_data_to_individuals_list(Exp, Experiment_ID)
        for indi in indi_list:
            sql_functions.insert_individual(indi, c)

    # Only commited after all trials run
    c.execute("COMMIT")
    conn.close()

approach_name = 'Parametric'
for environment_name, Exp_Environment in zip(env_names, exp_envs):
    conn = sqlite3.connect(data_base_address)
    c = conn.cursor()
    c.execute("BEGIN")
    most_recent = sql_functions.find_most_recent_experiment(conn)
    Experiment_ID = most_recent

    for repeat in tqdm(range(number_of_simulated_trials)):

        # Set Up Experiment ID and Random Seed
        Experiment_ID = Experiment_ID + 1
        np.random.seed(Experiment_ID)

        # Set up Models, Utility Function, and Selection Methods
        Efficacy_Model = Latent_Quadratic_Model(Exp_Environment.query_doses)
        Toxicity_Model = Null_Model(Exp_Environment.query_doses)
        Utility_Function = maximise_efficacy()
        Selection_Method = softmax(6.9)
        Final_Selection_Method = maximise_median()

        psuedo_x, psuedo_y = useful_functions.generate_pseudodata([0, .5, 1], [1, 5, 9], [9, 5, 1])
        Efficacy_Model.define_pseudo_data(psuedo_x, psuedo_y)
        Efficacy_Model.update_psuedo_weight(0.05)

        # Initialise experiment class
        Exp = Experiment(Exp_Environment, Efficacy_Model, Toxicity_Model,
                         Utility_Function, Selection_Method, Final_Selection_Method,
                         cohort_size)

        # Iterate for number of trial blocks
        Exp.full_loop(number_of_cohorts_per_trial)

        # Insert Experiment into SQL Queue
        exp_record = Experiment_Record(Experiment_ID, approach_name, environment_name)
        sql_functions.insert_experiment(exp_record, c)

        # Insert a row into the SQL queue for each individuals
        indi_list = sql_functions.experiment_data_to_individuals_list(Exp, Experiment_ID)
        for indi in indi_list:
            sql_functions.insert_individual(indi, c)

    # Only commited after all trials run
    c.execute("COMMIT")
    conn.close()

approach_name = 'Adaptive Naive'
for environment_name, Exp_Environment in zip(env_names, exp_envs):
    conn = sqlite3.connect(data_base_address)
    c = conn.cursor()
    c.execute("BEGIN")
    most_recent = sql_functions.find_most_recent_experiment(conn)
    Experiment_ID = most_recent

    for repeat in tqdm(range(number_of_simulated_trials)):

        # Set Up Experiment ID and Random Seed
        Experiment_ID = Experiment_ID + 1
        np.random.seed(Experiment_ID)

        # Set up Models, Utility Function, and Selection Methods
        Efficacy_Model = CoBe_Model(_d, kernel=uncorrelated_kernel())
        Toxicity_Model = Null_Model(_d)
        Utility_Function = maximise_efficacy()
        Selection_Method = thompson_sampling()
        Final_Selection_Method = maximise_median()

        # Initialise experiment class
        Exp = Experiment(Exp_Environment, Efficacy_Model, Toxicity_Model,
                         Utility_Function, Selection_Method, Final_Selection_Method,
                         cohort_size)

        # Iterate for number of trial blocks
        Exp.full_loop(number_of_cohorts_per_trial)

        # Insert Experiment into SQL Queue
        exp_record = Experiment_Record(Experiment_ID, approach_name, environment_name)
        sql_functions.insert_experiment(exp_record, c)

        # Insert a row into the SQL queue for each individuals
        indi_list = sql_functions.experiment_data_to_individuals_list(Exp, Experiment_ID)
        for indi in indi_list:
            sql_functions.insert_individual(indi, c)

    # Only commited after all trials run
    c.execute("COMMIT")
    conn.close()

approach_name = 'Uniform Naive'
for environment_name, Exp_Environment in zip(env_names, exp_envs):
    conn = sqlite3.connect(data_base_address)
    c = conn.cursor()
    c.execute("BEGIN")
    most_recent = sql_functions.find_most_recent_experiment(conn)
    Experiment_ID = most_recent

    for repeat in tqdm(range(number_of_simulated_trials)):

        # Set Up Experiment ID and Random Seed
        Experiment_ID = Experiment_ID + 1
        np.random.seed(Experiment_ID)

        # Set up Models, Utility Function, and Selection Methods
        Efficacy_Model = CoBe_Model(_d, kernel=uncorrelated_kernel())
        Toxicity_Model = Null_Model(_d)
        Utility_Function = maximise_efficacy()
        Selection_Method = uniform(_d)
        Final_Selection_Method = naive_best()

        # Initialise experiment class
        Exp = Experiment(Exp_Environment, Efficacy_Model, Toxicity_Model,
                         Utility_Function, Selection_Method, Final_Selection_Method,
                         cohort_size)

        # Iterate for number of trial blocks
        Exp.full_loop(number_of_cohorts_per_trial)

        # Insert Experiment into SQL Queue
        exp_record = Experiment_Record(Experiment_ID, approach_name, environment_name)
        sql_functions.insert_experiment(exp_record, c)

        # Insert a row into the SQL queue for each individuals
        indi_list = sql_functions.experiment_data_to_individuals_list(Exp, Experiment_ID)
        for indi in indi_list:
            sql_functions.insert_individual(indi, c)

    # Only commited after all trials run
    c.execute("COMMIT")
    conn.close()
