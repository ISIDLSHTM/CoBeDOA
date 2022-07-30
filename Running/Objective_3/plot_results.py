import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import pandas as pd
from matplotlib.ticker import (MultipleLocator)
from utility_functions import utility_contour

utility_function = utility_contour()

environments = ['Environment_1', 'Environment_2', 'Environment_3', 'Environment_3', 'Environment_5', 'Environment_6', ]
from environments import Environment_3_101_1, Environment_3_101_2, Environment_3_101_3, Environment_3_101_4, \
    Environment_3_441_5, Environment_3_441_6

scrape_environments = [Environment_3_101_1, Environment_3_101_2, Environment_3_101_3, Environment_3_101_4,
                       Environment_3_441_5, Environment_3_441_6]

linestyle = ["-",
             "-", "-",
             "-."]
colours = ['blue', 'orange', 'green', 'black']

env_map = {'Environment_1': 'Scenario 1',
           'Environment_2': 'Scenario 2',
           'Environment_3': 'Scenario 3',
           'Environment_3': 'Scenario 4',
           'Environment_5': 'Scenario 5',
           'Environment_6': 'Scenario 6'}

for i, env in enumerate(environments):
    if i < -1:
        pass
    else:
        print(env)
        conn = sqlite3.connect('__Objective_4_Database.db')
        c = conn.cursor()

        evironment_name = environments[i]

        c.execute("""SELECT individual_number, actual_utility_at_predicted, cumulative_regret, average_regret, 
                            method_name, efficacy, toxicity, experiments.experiment_id, inaccuracy
                            FROM individuals
                            INNER JOIN experiments
                             ON experiments.experiment_id = individuals.experiment_id
                             WHERE experiments.environment_name = ? 
                             AND experiments.method_name IN (
                             'CoBe Eff, CoBe Tox',
                              'Latent Quadratic Eff, Saturating Tox',
                              'Naive Uniform',
                              'Uncorrelated Thompson Sampling')""",
                  (evironment_name,), )
        joins = c.fetchall()

        joins = pd.DataFrame(joins)
        pd.set_option("display.max_columns", 999)
        pd.set_option("display.max_rows", 300)
        conn.close()
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        joins = joins.rename(columns={joins.columns[0]: 'individual_number',
                                      joins.columns[1]: 'actual_utility_at_predicted',
                                      joins.columns[2]: 'cumulative_regret',
                                      joins.columns[3]: 'average_regret',
                                      joins.columns[4]: 'method_name',
                                      joins.columns[5]: 'efficacy',
                                      joins.columns[6]: 'toxicity',
                                      joins.columns[7]: 'experiment_number',
                                      joins.columns[8]: 'inaccuracy'})

        joins['cum_eff'] = 0
        joins['cum_tox'] = 0
        joins['prop_eff'] = 0.0
        joins['prop_tox'] = 0.0

        exp_num = 0
        cum_eff = 0
        cum_tox = 0
        for index, row in joins.iterrows():
            if index % 10000 == 0:
                print(index / 180000)
            if row['experiment_number'] != exp_num:
                exp_num = row['experiment_number']
                cum_eff = 0
                cum_tox = 0
            cum_eff += row['efficacy']
            cum_tox += row['toxicity']
            joins.at[index, 'cum_eff'] = cum_eff
            joins.at[index, 'cum_tox'] = cum_tox
            joins.at[index, 'prop_eff'] = cum_eff / (row['individual_number'] + 1)
            joins.at[index, 'prop_tox'] = cum_tox / (row['individual_number'] + 1)
        joins_uti = joins
        joins_uti['utility'] = joins_uti.apply(
            lambda x: utility_function.get_dose_utility(np.array([[x['prop_eff']]]), np.array([[x['prop_tox']]]))[0],
            axis=1)
        joins_uti['scaled_utility'] = joins_uti.apply(
            lambda x: x['individual_number'] *
                      utility_function.get_dose_utility(np.array([[x['prop_eff']]]), np.array([[x['prop_tox']]]))[0],
            axis=1)

        joins = joins.loc[joins['individual_number'] % 6 == 5]

        _e = scrape_environments[i].efficacy_probabilities
        _t = scrape_environments[i].toxicity_probabilities
        _u = utility_function.get_dose_utility(_e, _t)

        best_index = np.argmax(_u)
        worst_index = np.argmin(_u)
        best_dose = scrape_environments[i].query_doses[best_index]
        worst_dose = scrape_environments[i].query_doses[worst_index]

        joins['individual_number'] += 1

        fig, ax = plt.subplots(figsize=(15, 7))
        _joins = joins[['method_name', 'individual_number', 'actual_utility_at_predicted']]
        for midx, _method in enumerate(['CoBe', 'Parametric', 'Adaptive Naive', 'Uniform Naive']):
            print(_method)
            df = pd.DataFrame(columns=['individual_number', 'mean', 'lower', 'upper'])
            __joins = _joins.loc[_joins['method_name'] == _method]
            __joins = __joins[__joins.individual_number % 6 == 0]
            for _individual in __joins['individual_number'].unique():
                ___joins = __joins.loc[__joins['individual_number'] == _individual]

                _mean = ___joins['actual_utility_at_predicted'].mean()
                _std = ___joins['actual_utility_at_predicted'].std() / np.sqrt(___joins.shape[0])
                _upper = min(max(_u), _mean + 1.96 * _std)
                _lower = max(min(_u), _mean - 1.96 * _std)

                df = df.append({'individual_number': _individual,
                                'mean': _mean,
                                'lower': _lower,
                                'upper': _upper}, ignore_index=True)
            plt.plot(df['individual_number'], df['mean'], color=colours[midx], linestyle=linestyle[midx], label=_method)
            plt.fill_between(df['individual_number'], df['lower'], df['upper'], color=colours[midx], alpha=0.1)

        plt.legend(title='Dose-Optimisation Approach', loc=4, fontsize=15, title_fontsize=15)
        plt.title(env_map[env], fontsize=15)
        plt.tight_layout(rect=[0.015, 0.01, .99, .99])
        plt.xticks([0, 6, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300], fontsize=12)
        ax.xaxis.set_minor_locator(MultipleLocator(6))
        plt.yticks(fontsize=12)
        plt.xlabel('Number Of Trial Participants', fontsize=15)
        plt.ylabel('Mean true utility at predicted optimal dose', fontsize=15)
        plt.ylim(min(_u) - (max(_u) - min(_u)) / 12, max(_u) + (max(_u) - min(_u)) / 12)
        plt.plot([6, 300], [min(_u), min(_u)], c='brown')
        plt.plot([6, 300], [max(_u), max(_u)], c='brown')
        plt.xlim(6, 300)
        address = 'Plots/' + env + '_Utility.png'
        plt.savefig(address)
        plt.close()

        fig, ax = plt.subplots(figsize=(15, 7))
        _joins = joins_uti[['method_name', 'individual_number', 'scaled_utility']]
        for midx, _method in enumerate(['CoBe', 'Parametric', 'Adaptive Naive', 'Uniform Naive']):
            print(_method)
            df = pd.DataFrame(columns=['individual_number', 'mean', 'lower', 'upper'])
            __joins = _joins.loc[_joins['method_name'] == _method]

            for _individual in __joins['individual_number'].unique():
                ___joins = __joins.loc[__joins['individual_number'] == _individual]

                _mean = ___joins['scaled_utility'].mean()
                _std = ___joins['scaled_utility'].std() / np.sqrt(___joins.shape[0])
                _upper = min(max(_u) * _individual, _mean + 1.96 * _std)
                _lower = max(min(_u) * _individual, _mean - 1.96 * _std)

                df = df.append({'individual_number': _individual,
                                'mean': _mean,
                                'lower': _lower,
                                'upper': _upper}, ignore_index=True)
            plt.plot(df['individual_number'], df['mean'], color=colours[midx], linestyle=linestyle[midx], label=_method)
            plt.fill_between(df['individual_number'], df['lower'], df['upper'], color=colours[midx], alpha=0.1)

        plt.legend(title='Dose-Optimisation Approach', loc=4, fontsize=15, title_fontsize=15)
        plt.title(env_map[env], fontsize=15)
        plt.tight_layout(rect=[0.015, 0.01, .99, .99])
        plt.xticks([0, 6, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300], fontsize=12)
        ax.xaxis.set_minor_locator(MultipleLocator(6))
        plt.yticks(fontsize=12)
        plt.xlabel('Number Of Trial Participants', fontsize=15)
        plt.ylabel('Mean Cumulative Utility', fontsize=15)
        plt.plot([0, 300], [0, 300 * min(_u)], c='brown')
        plt.plot([0, 300], [0, 300 * max(_u)], c='brown')
        plt.ylim(min(0, 300 * min(_u)), 300 * max(_u))
        plt.xlim(0, 300)
        address = 'Plots/' + env + '_CumulativeUtility.png'
        plt.savefig(address)
        plt.close()
