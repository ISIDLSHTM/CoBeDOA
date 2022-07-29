import numpy as np
from sql_functions_and_classes.sql_classes import *


def insert_experiment(exp, c):
    c.execute("""INSERT INTO experiments VALUES(
            :experiment_id,
            :method_name,
            :environment_name)""", {'experiment_id': exp.Experiment_ID,
                                    'method_name': exp.Method_Name,
                                    'environment_name': exp.Environment_Name})


def insert_individual(ind, c):
    c.execute("""INSERT INTO individuals VALUES(
            :experiment_id,
            :individual_number,
            :dose,
            :efficacy,
            :toxicity,
            :predicted_optimal_dose,
            :actual_utility_at_predicted,
            :inaccuracy,
            :individual_regret,
            :cumulative_regret,
            :average_regret)""",
              {'experiment_id': ind.Experiment_ID,
               'individual_number': ind.Individual_Number,
               'dose': ind.Dose,
               'efficacy': ind.Efficacy,
               'toxicity': ind.Toxicity,
               'predicted_optimal_dose': ind.Predicted_Optimal_Dose,
               'actual_utility_at_predicted': np.float(ind.Actual_Utility_At_Predicted),
               'inaccuracy': ind.Inaccuracy,
               'individual_regret': ind.Individual_Regret,
               'cumulative_regret': ind.Cumulative_Regret,
               'average_regret': ind.Average_Regret
               })


def find_most_recent_experiment(connection):
    with connection:  # stops us needing to commit
        c = connection.cursor()
        c.execute('SELECT max(rowid) FROM experiments')
        most_recent_id = c.fetchone()[0]
    if most_recent_id is None:
        return 0
    return most_recent_id


def experiment_data_to_individuals_list(Exp, Experiment_ID):
    indi_list = []
    for i in range(len(Exp.dose_data)):
        Individual_Number = i
        _Dose = Exp.dose_data[i]
        Dose = ''
        first = True
        for d in _Dose:
            if first:
                Dose = Dose + str(d)
            else:
                Dose = Dose + '_' + str(d)
            first = False

        Efficacy = str(Exp.efficacy_data[i])
        Toxicity = str(Exp.toxicity_data[i])

        _Predicted_Optimal_Dose = Exp.predicted_optimal_doses[i]
        Predicted_Optimal_Dose = ''
        first = True
        for d in _Predicted_Optimal_Dose:
            if first:
                Predicted_Optimal_Dose = Predicted_Optimal_Dose + str(d)
            else:
                Predicted_Optimal_Dose = Predicted_Optimal_Dose + '_' + str(d)
            first = False

        Predicted_Utility = Exp.predicted_optimal_utilities[i]
        Actual_Utility_At_Predicted = Exp.actual_utility_at_predicted_optimal[i]
        Individual_Regret = Exp.individual_regrets[i]
        Cumulative_Regret = Exp.cumulative_regrets[i]
        Average_Regret = Exp.average_regrets[i]
        Inaccuracy = Predicted_Utility - Actual_Utility_At_Predicted
        indi = Individual(Experiment_ID, Individual_Number,
                          Dose, Efficacy, Toxicity,
                          Predicted_Optimal_Dose, Predicted_Utility, Actual_Utility_At_Predicted, Inaccuracy,
                          Individual_Regret, Cumulative_Regret, Average_Regret)
        indi_list.append(indi)

    return indi_list
