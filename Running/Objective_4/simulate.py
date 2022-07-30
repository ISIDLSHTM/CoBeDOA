from utility_functions import maximise_efficacy, utility_contour
from models import CoBe_Model, Null_Model
from experiment_class import Experiment
from selection_methods import thompson_sampling
from final_selection_methods import maximise_median
from Functions_for_modelling.kernels import squared_exponential_kernel
import useful_functions

from environments import Environment_1_101_1, Environment_1_101_4, Environment_2_441_1, Environment_2_441_2, \
    Environment_2_1331_6, Environment_2_1331_7, Environment_3_441_5

from sql_functions_and_classes.sql_classes import *
from sql_functions_and_classes import sql_functions

import sqlite3
import numpy as np
from tqdm import tqdm
from config import CFG

number_of_simulated_trials = CFG.num_sims
data_base_address = '__Objective_4_Database.db'

block_size = 6
number_of_blocks_per_trial = 50

env_names = ['Environment_1', 'Environment_2', 'Environment_3',
             'Environment_4', 'Environment_5', 'Environment_6']
exp_envs = [Environment_1_101_1, Environment_1_101_4, Environment_2_441_1, Environment_2_441_2,
            Environment_2_1331_6, Environment_2_1331_7]
kernel_lengths = [.2, .2, .25, .25, .4, .4]
approach_names = ['Very Strong, Correct',
                  'Strong, Correct',
                  'No Prior',
                  'Strong, Incorrect',
                  'Very Strong, Incorrect']
confidences = [20, 3, 0, 3, 20]
correctnesss = [1, 1, 0, -1, -1]
for enviroment_name, Exp_Environment, kernel_length in zip(env_names, exp_envs, kernel_lengths):
    for approach_name, confidence, correctness in zip(approach_names, confidences, correctnesss):

        conn = sqlite3.connect(data_base_address)
        c = conn.cursor()
        c.execute("BEGIN")

        most_recent = sql_functions.find_most_recent_experiment(conn)
        Experiment_ID = most_recent
        print(enviroment_name, Exp_Environment, approach_name, block_size, number_of_blocks_per_trial)
        print('With starting Experiment ID ', Experiment_ID + 1, ', we are simulating '
              , number_of_simulated_trials, ' trials, each containing ',
              number_of_blocks_per_trial * block_size, ' individuals divided into ', number_of_blocks_per_trial,
              ' blocks of ', block_size, '.', sep='')

        for repeat in tqdm(range(number_of_simulated_trials)):

            # Set Up Experiment ID and Random Seed
            Experiment_ID = Experiment_ID + 1
            np.random.seed(Experiment_ID)

            # Set up Models, Utility Function, and Selection Methods
            Efficacy_Model = CoBe_Model(Exp_Environment.query_doses, kernel=squared_exponential_kernel(kernel_length))
            Toxicity_Model = Null_Model(Exp_Environment.query_doses)
            Utility_Function = maximise_efficacy()
            Selection_Method = thompson_sampling()
            Final_Selection_Method = maximise_median()

            if correctness == 1:
                expected = Exp_Environment.efficacy_probabilities.reshape(-1)
                confidence = confidence + (0 * expected)
                prior = useful_functions.beta_mode_array_to_parameters(expected, confidence)
                Efficacy_Model.define_prior_and_setup(prior)
            elif correctness == -1:
                expected = Exp_Environment.efficacy_probabilities.reshape(-1)
                expected = 1 - expected
                confidence = confidence + (0 * expected)
                prior = useful_functions.beta_mode_array_to_parameters(expected, confidence)
                Efficacy_Model.define_prior_and_setup(prior)

            # Initialise experiment class
            Exp = Experiment(Exp_Environment, Efficacy_Model, Toxicity_Model,
                             Utility_Function, Selection_Method, Final_Selection_Method,
                             block_size)

            # Iterate for number of trial blocks
            Exp.full_loop(number_of_blocks_per_trial)

            # Insert Experiment into SQL Queue
            exp_record = Experiment_Record(Experiment_ID, approach_name, enviroment_name)
            sql_functions.insert_experiment(exp_record, c)

            # Insert a row into the SQL queue for each individuals
            indi_list = sql_functions.experiment_data_to_individuals_list(Exp, Experiment_ID)
            for indi in indi_list:
                sql_functions.insert_individual(indi, c)

        # Only commited after all trials run
        c.execute("COMMIT")
        conn.close()

env_names = ['Environment_7']
exp_envs = [Environment_3_441_5]
for enviroment_name, Exp_Environment in zip(env_names, exp_envs):
    for approach_name, confidence, correctness in zip(approach_names, confidences, correctnesss):
        conn = sqlite3.connect(data_base_address)
        c = conn.cursor()
        c.execute("BEGIN")

        most_recent = sql_functions.find_most_recent_experiment(conn)
        Experiment_ID = most_recent
        print(enviroment_name, Exp_Environment, approach_name, block_size, number_of_blocks_per_trial)
        print('With starting Experiment ID ', Experiment_ID + 1, ', we are simulating '
              , number_of_simulated_trials, ' trials, each containing ',
              number_of_blocks_per_trial * block_size, ' individuals divided into ', number_of_blocks_per_trial,
              ' blocks of ', block_size, '.', sep='')

        for repeat in tqdm(range(number_of_simulated_trials)):

            # Set Up Experiment ID and Random Seed
            Experiment_ID = Experiment_ID + 1
            np.random.seed(Experiment_ID)

            # Set up Models, Utility Function, and Selection Methods
            Efficacy_Model = CoBe_Model(Exp_Environment.query_doses, kernel=squared_exponential_kernel(.25))
            Toxicity_Model = CoBe_Model(Exp_Environment.query_doses, kernel=squared_exponential_kernel(.25))
            Utility_Function = utility_contour()
            Selection_Method = thompson_sampling()
            Final_Selection_Method = maximise_median()

            if correctness == 1:
                expected = Exp_Environment.efficacy_probabilities.reshape(-1)
                confidence = confidence + (0 * expected)
                prior = useful_functions.beta_mode_array_to_parameters(expected, confidence)
                Efficacy_Model.define_prior_and_setup(prior)
            elif correctness == -1:
                expected = Exp_Environment.efficacy_probabilities.reshape(-1)
                expected = 1 - expected
                confidence = confidence + (0 * expected)
                prior = useful_functions.beta_mode_array_to_parameters(expected, confidence)
                Efficacy_Model.define_prior_and_setup(prior)

            # Initialise experiment class
            Exp = Experiment(Exp_Environment, Efficacy_Model, Toxicity_Model,
                             Utility_Function, Selection_Method, Final_Selection_Method,
                             block_size)

            # Iterate for number of trial blocks
            Exp.full_loop(number_of_blocks_per_trial)

            # Insert Experiment into SQL Queue
            exp_record = Experiment_Record(Experiment_ID, approach_name, enviroment_name)
            sql_functions.insert_experiment(exp_record, c)

            # Insert a row into the SQL queue for each individuals
            indi_list = sql_functions.experiment_data_to_individuals_list(Exp, Experiment_ID)
            for indi in indi_list:
                sql_functions.insert_individual(indi, c)

        # Only commited after all trials run
        c.execute("COMMIT")
        conn.close()
