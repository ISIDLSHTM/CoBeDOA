import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import numpy as np

from matplotlib.ticker import MultipleLocator

from environments import Environment_1_101_1, Environment_1_101_2, Environment_1_101_3, Environment_1_101_4, \
    Environment_1_101_5, Environment_1_101_6, Environment_1_101_7

methods_subset = []

env_map = {'Environment_1': 'Scenario 1',
           'Environment_2': 'Scenario 2',
           'Environment_3': 'Scenario 3',
           'Environment_4': 'Scenario 4',
           'Environment_5': 'Scenario 5',
           'Environment_6': 'Scenario 6',
           'Environment_7': 'Scenario 7'}

environments = ['Environment_1', 'Environment_2', 'Environment_3', 'Environment_4', 'Environment_5', 'Environment_6',
                'Environment_7']
scrape_environments = [Environment_1_101_1, Environment_1_101_2, Environment_1_101_3, Environment_1_101_4,
                       Environment_1_101_5, Environment_1_101_6, Environment_1_101_7]

linestyle = ["-",
             "-", "-",
             "-."]
colours = ['blue', 'orange', 'green', 'black']

for i, env in enumerate(environments):
    print(env)
    conn = sqlite3.connect('__Objective_1_Database.db')
    c = conn.cursor()

    evironment_name = environments[i]

    c.execute("""SELECT individual_number, actual_utility_at_predicted, cumulative_regret, average_regret, method_name, efficacy, inaccuracy, experiments.experiment_id
                    FROM individuals
                    INNER JOIN experiments
                     ON experiments.experiment_id = individuals.experiment_id
                     WHERE experiments.environment_name = ?""", (evironment_name,), )
    joins = c.fetchall()

    joins = pd.DataFrame(joins)
    conn.close()
    joins = joins.rename(columns={joins.columns[0]: 'individual_number',
                                  joins.columns[1]: 'actual_utility_at_predicted',
                                  joins.columns[2]: 'cumulative_regret',
                                  joins.columns[3]: 'average_regret',
                                  joins.columns[4]: 'method_name',
                                  joins.columns[5]: 'efficacy',
                                  joins.columns[6]: 'inaccuracy',
                                  joins.columns[7]: 'experiment_id'})

    joins_eff = joins

    _u = scrape_environments[i].efficacy_probabilities

    best_index = np.argmax(_u)
    worst_index = np.argmin(_u)
    best_dose = scrape_environments[i].query_doses[best_index]
    worst_dose = scrape_environments[i].query_doses[worst_index]
    joins['individual_number'] += 1

    fig, ax = plt.subplots(figsize=(15, 10))
    _joins = joins[['method_name', 'individual_number', 'actual_utility_at_predicted']]
    for midx, _method in enumerate(['CoBe', 'Parametric', 'Adaptive Naive', 'Uniform Naive']):
        df = pd.DataFrame(columns=['individual_number', 'mean', 'lower', 'upper'])
        __joins = _joins.loc[_joins['method_name'] == _method]
        __joins = __joins[__joins.individual_number % 6 == 0]
        for _individual in __joins['individual_number'].unique():
            ___joins = __joins.loc[__joins['individual_number'] == _individual]

            _mean = ___joins['actual_utility_at_predicted'].mean()
            _std = ___joins['actual_utility_at_predicted'].std() / np.sqrt(___joins.shape[0])
            _upper = min(1, _mean + 1.96 * _std)
            _lower = max(0, _mean - 1.96 * _std)

            df = df.append({'individual_number': _individual,
                            'mean': _mean,
                            'lower': _lower,
                            'upper': _upper}, ignore_index=True)
        plt.plot(df['individual_number'], df['mean'], color=colours[midx], linestyle=linestyle[midx], label=_method,
                 linewidth=3)
        plt.fill_between(df['individual_number'], df['lower'], df['upper'], color=colours[midx], alpha=0.1)

    plt.legend(title='Dose-Optimisation Approach', loc=4, fontsize=25, title_fontsize=25)
    plt.title('Objective 1, ' + env_map[env], fontsize=28)
    plt.tight_layout(rect=[0.05, 0.05, .99, .97])
    plt.xticks([0, 6, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300], fontsize=22)
    ax.xaxis.set_minor_locator(MultipleLocator(6))
    plt.yticks(fontsize=22)
    plt.xlabel('Number Of Trial Participants', fontsize=25)
    plt.ylabel('Mean true efficacy at predicted optimal dose', fontsize=25)
    plt.ylim(min(_u) - (max(_u) - min(_u)) / 24, max(_u) + (max(_u) - min(_u)) / 24)
    plt.plot([6, 300], [min(_u), min(_u)], c='brown', linewidth=3)
    plt.plot([6, 300], [max(_u), max(_u)], c='brown', linewidth=3)
    ax.xaxis.set_tick_params(width=5, length=5)
    ax.yaxis.set_tick_params(width=5, length=5)
    plt.xlim(6, 300)
    address = 'Plots/' + env + '_Utility.png'
    plt.savefig(address)
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 10))
    _joins = joins[['method_name', 'individual_number', 'efficacy']]
    for midx, _method in enumerate(['CoBe', 'Parametric', 'Adaptive Naive', 'Uniform Naive']):
        df = pd.DataFrame(columns=['individual_number', 'mean', 'lower', 'upper'])
        __joins = _joins.loc[_joins['method_name'] == _method]
        cumsum = None
        for _individual in __joins['individual_number'].unique():
            ___joins = __joins.loc[__joins['individual_number'] == _individual]
            if cumsum is None:
                cumsum = ___joins['efficacy'].to_numpy()
            else:
                cumsum += ___joins['efficacy'].to_numpy()

            _mean = cumsum.mean()
            _std = np.std(cumsum) / np.sqrt(np.size(cumsum))
            _upper = _mean + 1.96 * _std
            _lower = _mean - 1.96 * _std

            df = df.append({'individual_number': _individual,
                            'mean': _mean,
                            'lower': _lower,
                            'upper': _upper}, ignore_index=True)
        plt.plot(df['individual_number'], df['mean'], color=colours[midx], linestyle=linestyle[midx], label=_method,
                 linewidth=3)
        plt.fill_between(df['individual_number'], df['lower'], df['upper'], color=colours[midx], alpha=0.1)

    plt.legend(title='Dose-Optimisation Approach', loc=4, fontsize=25, title_fontsize=25)
    plt.title('Objective 1, ' + env_map[env], fontsize=28)
    plt.tight_layout(rect=[0.05, 0.05, .99, .97])
    plt.xticks([0, 6, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300], fontsize=22)
    ax.xaxis.set_minor_locator(MultipleLocator(6))
    plt.yticks(fontsize=22)
    plt.xlabel('Number Of Trial Participants', fontsize=25)
    plt.ylabel('Mean Cumulative Sum of Efficacy', fontsize=25)
    plt.plot([0, 300], [0, 300 * min(_u)], c='brown', linewidth=3)
    plt.plot([0, 300], [0, 300 * max(_u)], c='brown', linewidth=3)
    ax.xaxis.set_tick_params(width=5, length=5)
    ax.yaxis.set_tick_params(width=5, length=5)
    plt.ylim(0, 300 * max(_u))
    plt.xlim(0, 300)
    address = 'Plots/' + env + '_CumulativeUtility.png'
    plt.savefig(address)
    plt.close()
