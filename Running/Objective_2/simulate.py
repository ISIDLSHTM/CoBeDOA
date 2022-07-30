from utility_functions import maximise_efficacy
from models import CoBe_Model, Null_Model, Latent_Quadratic_Model_2D, Latent_Quadratic_Model_3D
from experiment_class import Experiment
from selection_methods import thompson_sampling, softmax, uniform
from final_selection_methods import maximise_median, naive_best
from Functions_for_modelling.kernels import squared_exponential_kernel, uncorrelated_kernel

from environments import Environment_2_441_1, Environment_2_441_2, Environment_2_441_3, \
    Environment_2_441_4, Environment_2_441_5, Environment_2_1331_6, Environment_2_1331_7
from sql_functions_and_classes.sql_classes import *
from sql_functions_and_classes import sql_functions

import sqlite3
import numpy as np
from tqdm import tqdm
from config import CFG
import useful_functions

number_of_simulated_trials = CFG.num_sims
data_base_address = '__Objective_2_Database.db'

cohort_size = 6
number_of_cohorts_per_trial = 50

env_names = ['Environment_1', 'Environment_2', 'Environment_3',
             'Environment_4', 'Environment_5', 'Environment_6',
             'Environment_7']
exp_envs = [Environment_2_441_1, Environment_2_441_2, Environment_2_441_3, \
            Environment_2_441_4, Environment_2_441_5, Environment_2_1331_6, Environment_2_1331_7]
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
        Efficacy_Model = CoBe_Model(Exp_Environment.query_doses, kernel=squared_exponential_kernel(.25))
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

_d = np.array([[0, 0], [.5, 0], [1, 0],
               [0, .5], [.5, .5], [1, .5],
               [0, 1], [.5, 1], [1, 1]]).reshape(-1, 2)
env_names = ['Environment_1', 'Environment_2', 'Environment_3',
             'Environment_4', 'Environment_5']
exp_envs = [Environment_2_441_1, Environment_2_441_2, Environment_2_441_3, \
            Environment_2_441_4, Environment_2_441_5]

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

cohort_size = 9
number_of_cohorts_per_trial = 33

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

_d = np.array([[0, 0, 0], [0, .5, 0], [0, 1, 0],
               [0, 0, .5], [0, .5, .5], [0, 1, .5],
               [0, 0, 1], [0, .5, 1], [0, 1, 1],

               [0.5, 0, 0], [0.5, .5, 0], [0.5, 1, 0],
               [0.5, 0, .5], [0.5, .5, .5], [0.5, 1, .5],
               [0.5, 0, 1], [0.5, .5, 1], [0.5, 1, 1],

               [1, 0, 0], [1, .5, 0], [1, 1, 0],
               [1, 0, .5], [1, .5, .5], [1, 1, .5],
               [1, 0, 1], [1, .5, 1], [1, 1, 1]
               ]).reshape(-1, 3)
env_names = ['Environment_6',
             'Environment_7']
exp_envs = [Environment_2_1331_6, Environment_2_1331_7]

cohort_size = 27
number_of_cohorts_per_trial = 11
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

cohort_size = 6
number_of_cohorts_per_trial = 50
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

env_names = ['Environment_1', 'Environment_2', 'Environment_3',
             'Environment_4', 'Environment_5']
exp_envs = [Environment_2_441_1, Environment_2_441_2, Environment_2_441_3, \
            Environment_2_441_4, Environment_2_441_5]

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
        Efficacy_Model = Latent_Quadratic_Model_2D(Exp_Environment.query_doses)
        Toxicity_Model = Null_Model(Exp_Environment.query_doses)
        Utility_Function = maximise_efficacy()
        Selection_Method = softmax(6.9)
        Final_Selection_Method = maximise_median()

        psuedo_x, psuedo_y = useful_functions.generate_pseudodata_2D([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]],
                                                                     [1, 5, 5, 9, 5], [9, 5, 5, 1, 5])
        Efficacy_Model.define_pseudo_data(psuedo_x, psuedo_y)
        Efficacy_Model.update_psuedo_weight(0.03)

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

env_names = ['Environment_6',
             'Environment_7']
exp_envs = [Environment_2_1331_6, Environment_2_1331_7]

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
        Efficacy_Model = Latent_Quadratic_Model_3D(Exp_Environment.query_doses)
        Toxicity_Model = Null_Model(Exp_Environment.query_doses)
        Utility_Function = maximise_efficacy()
        Selection_Method = softmax(6.9)
        Final_Selection_Method = maximise_median()

        psuedo_x, psuedo_y = useful_functions.generate_pseudodata_3D([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                                                      [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5],
                                                                      [1, 1, 1]],
                                                                     [1, 5, 5, 5, 5, 5, 5, 5, 9],
                                                                     [9, 5, 5, 5, 5, 5, 5, 5, 1])
        Efficacy_Model.define_pseudo_data(psuedo_x, psuedo_y)
        Efficacy_Model.update_psuedo_weight(1.5 / 90)

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
