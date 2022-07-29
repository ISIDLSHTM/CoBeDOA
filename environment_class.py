import numpy as np
import sys
import useful_functions as uf
import matplotlib.pyplot as plt
import warnings
import copy


class Environment():
    def __init__(self, query_doses,
                 efficacy_probabilities,
                 toxicity_probabilities):
        self.query_doses = copy.deepcopy(query_doses)  # shape is (number of query doses, number of features)
        self.efficacy_probabilities = copy.deepcopy(
            efficacy_probabilities)  # shape is (number of query doses, outputsize)
        self.toxicity_probabilities = copy.deepcopy(
            toxicity_probabilities)  # shape is (number of query doses, outputsize)
        self.ordinal_setup()
        self.sanity_check()

    def sanity_check(self):
        if (np.shape(self.query_doses)[0] != np.shape(self.efficacy_probabilities)[0]):
            sys.exit('asdcxdv')
        if (np.shape(self.query_doses)[0] != np.shape(self.toxicity_probabilities)[0]):
            sys.exit('asdcxdvdb')

        if self.ordinal_efficacy:
            sums = np.sum(self.efficacy_probabilities, axis=1)
            if np.any(sums != 1):
                sys.exit('cmmmc opc')
        if self.ordinal_toxicity:
            sums = np.sum(self.toxicity_probabilities, axis=1)
            if np.any(sums != 1):
                sys.exit('cmmmcdddd opc')

    def ordinal_setup(self):
        if np.shape(self.efficacy_probabilities)[1] == 1:
            self.ordinal_efficacy = False
        else:
            self.ordinal_efficacy = True

        if np.shape(self.toxicity_probabilities)[1] == 1:
            self.ordinal_toxicity = False
        else:
            self.ordinal_toxicity = True

    def get_dose_efficacy_probability_non_ord(self, doses=None):
        if doses == None:
            doses = self.query_doses

        arguments = []
        for d in doses:
            argument = np.where(np.isclose(self.query_doses, d).all(axis=1))[0]
            arguments.append(argument)
        arguments = np.asarray(arguments).reshape(-1)

        return self.efficacy_probabilities[arguments].reshape(-1)

    def get_dose_toxicity_probability_non_ord(self, doses=None):
        if doses == None:
            doses = self.query_doses
        arguments = []
        for d in doses:
            argument = np.where(np.isclose(self.query_doses, d).all(axis=1))[0]
            arguments.append(argument)
        arguments = np.asarray(arguments).reshape(-1)
        return self.toxicity_probabilities[arguments].reshape(-1)

    def get_dose_efficacy_probability_ord(self, doses=None):
        if doses == None:
            doses = self.query_doses
        if not np.all(np.isin(doses, self.query_doses)):
            warnings.warn('This is a default warning.')

        arguments = []
        for d in doses:
            argument = np.where((self.query_doses == d).all(axis=1))[0]
            arguments.append(argument)
        arguments = np.asarray(arguments).reshape(-1)
        return self.efficacy_probabilities[arguments, :]

    def get_dose_toxicity_probability_ord(self, doses=None):
        if doses == None:
            doses = self.query_doses
        if not np.all(np.isin(doses, self.query_doses)):
            warnings.warn('This is a default warning.')

        arguments = []
        for d in doses:
            argument = np.where((self.query_doses == d).all(axis=1))[0]
            arguments.append(argument)
        arguments = np.asarray(arguments).reshape(-1)
        return self.toxicity_probabilities[arguments, :]

    def sample_dose_efficacy_non_ord(self, doses):
        probabilities = self.get_dose_efficacy_probability_non_ord(doses)
        samples = uf.sample_from_probability(probabilities)
        return samples

    def sample_dose_toxicity_non_ord(self, doses):
        probabilities = self.get_dose_toxicity_probability_non_ord(doses)
        samples = uf.sample_from_probability(probabilities)
        return samples

    def sample_dose_efficacy_ord(self, doses):
        probabilities = self.get_dose_efficacy_probability_ord(doses)
        samples = uf.sample_from_ordinal_probability(probabilities)
        return samples

    def sample_dose_toxicity_ord(self, doses):
        probabilities = self.get_dose_toxicity_probability_ord(doses)
        samples = uf.sample_from_ordinal_probability(probabilities)
        return samples

    def sample_dose_efficacy(self, doses=None):
        if not self.ordinal_efficacy:
            return self.sample_dose_efficacy_non_ord(doses)
        return self.sample_dose_efficacy_ord(doses)

    def sample_dose_toxicity(self, doses=None):
        if not self.ordinal_toxicity:
            return self.sample_dose_toxicity_non_ord(doses)
        return self.sample_dose_toxicity_ord(doses)

    def plot_efficacy(self):
        plt.plot(self.query_doses, self.efficacy_probabilities)
        plt.show()
