import numpy as np
import sys
import useful_functions as uf


class thompson_sampling():
    def __init__(self):
        pass

    def get_next_doses(self, environment, efficacy_model, toxicity_model, utility_function, block_size):
        next_doses = []
        for i in range(block_size):
            predicted_efficacy = efficacy_model.get_prediction()
            predicted_toxicity = toxicity_model.get_prediction()
            doses, predicted_efficacy = predicted_efficacy
            _doses, predicted_toxicity = predicted_toxicity
            if not np.array_equal(doses, _doses):
                sys.exit('ts: Something went horribly wrong 1')
            # if not np.array_equal(doses, environment.query_doses):
            #     sys.exit('ts: Something went horribly wrong 2')
            predicted_utilities = utility_function.get_dose_utility(predicted_efficacy, predicted_toxicity)
            best_arg = uf.randargmax(predicted_utilities)
            best_dose = doses[best_arg]
            next_doses.append(best_dose)
        return next_doses


class softmax():
    def __init__(self, inverse_temperature):
        self.inverse_temperature = inverse_temperature
        pass

    def get_next_doses(self, environment, efficacy_model, toxicity_model, utility_function, block_size):
        next_doses = []
        predicted_efficacy = efficacy_model.get_prediction()
        predicted_toxicity = toxicity_model.get_prediction()
        doses, predicted_efficacy = predicted_efficacy
        _doses, predicted_toxicity = predicted_toxicity
        if not np.array_equal(doses, _doses):
            sys.exit('Something went horribly wrong')
        # if not np.array_equal(doses, environment.query_doses):
        #     sys.exit('Something went horribly wrong')
        predicted_utilities = utility_function.get_dose_utility(predicted_efficacy, predicted_toxicity)

        chosen_args = uf.soft_max_selector(predicted_utilities, self.inverse_temperature, block_size)
        for arg in chosen_args:
            best_dose = doses[arg]
            next_doses.append(best_dose)
        return next_doses


class uniform():
    def __init__(self, doses_to_choose):
        next_doses = []
        for dose in doses_to_choose:
            # next_doses.append(np.array([float(dose)]))
            next_doses.append(np.array(dose).astype(float))
        self.next_doses = next_doses
        pass

    def get_next_doses(self, environment, efficacy_model, toxicity_model, utility_function, block_size):
        if len(self.next_doses) == block_size:
            return self.next_doses
        sys.exit('Block size and design not consistent')
