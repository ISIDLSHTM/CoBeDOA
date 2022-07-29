import numpy as np
import sys
import useful_functions as uf


class maximise_median():
    def __init__(self):
        pass

    def predict_optimal(self, environment, efficacy_model, toxicity_model, utility_function, block_size):
        predicted_efficacy = efficacy_model.get_prediction(random=False, desired_percentile=.5)
        predicted_toxicity = toxicity_model.get_prediction(random=False, desired_percentile=.5)
        doses, predicted_efficacy = predicted_efficacy
        _doses, predicted_toxicity = predicted_toxicity
        if not np.array_equal(doses, _doses):
            sys.exit('fsm: Something went horribly wrong 1, maximise median')
        if not np.array_equal(doses, environment.query_doses):
            pass
            # print(doses, environment.query_doses)
            # print('fsm: Careful, doses and environment.query_doses are not equal')
        predicted_utilities = utility_function.get_dose_utility(predicted_efficacy, predicted_toxicity)
        best_arg = uf.randargmax(predicted_utilities)
        best_dose = doses[best_arg]
        best_utility = predicted_utilities[best_arg]
        best_arg = 0
        for i in environment.query_doses:
            if np.allclose(i, best_dose):
                break
            best_arg += 1
        return best_dose, best_utility, doses, best_arg


class naive_best():
    def __init__(self):
        pass

    def predict_optimal(self, environment, efficacy_model, toxicity_model, utility_function, block_size):
        predicted_efficacy = efficacy_model.get_prediction(random=False, desired_percentile=.5)
        predicted_toxicity = toxicity_model.get_prediction(random=False, desired_percentile=.5)
        doses, predicted_efficacy = predicted_efficacy
        _doses, predicted_toxicity = predicted_toxicity
        if not np.array_equal(doses, _doses):
            sys.exit('fsm: Something went horribly wrong 1')
        predicted_utilities = utility_function.get_dose_utility(predicted_efficacy, predicted_toxicity)
        best_arg = uf.randargmax(predicted_utilities)
        best_dose = doses[best_arg]
        best_utility = predicted_utilities[best_arg]
        best_arg = 0
        for i in environment.query_doses:
            if np.allclose(i, best_dose):
                break
            best_arg += 1
        return best_dose, best_utility, doses, best_arg
