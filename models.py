import Functions_for_modelling.kernels as k
import sys
import useful_functions

from Functions_for_modelling.parametric_model_functions import *
from scipy.stats import beta

class CoBe_Model():
    def __init__(self, feature_array, number_of_levels=2, kernel=k.uncorrelated_kernel(), data=None, scalar=None):
        self.features = feature_array
        self.kernel = kernel  # the similarity kernel
        self.efficient_kernel = kernel.efficient
        self.number_of_levels = number_of_levels  # Levels are 0,1,...,n-1 where n = number of levels
        self.get_number_of_feature_rows_and_columns()
        self.create_flat_priors(scalar)
        self.init_data()
        self.init_likelihood()
        if data is not None:
            self.add_data(data)  # data is a tuple of (features, values),
            # with features being a (number of samples, number of feature columns)-nparray and
            # values being a (number of samples, 1)-nparray containing integers in [0,number of levels 1]
            self.update_likelihood()
        self.update_posterior()

    def add_data(self, data):
        data_features, data_values = data
        if (data_values >= 0).all() \
                and (data_values < self.number_of_levels).all() \
                and useful_functions.isinteger(data_values).all():
            self.data_features = np.concatenate((self.data_features, data_features))
            self.data_values = np.concatenate((self.data_values, data_values))
            self.new_data_features = data_features
            self.new_data_values = data_values
        else:
            sys.exit('Error in add_data: All values should be integers and within the number of potential levels')

    def create_flat_priors(self, scalar=None):
        """
        Creates an attribute 'prior'.
        This is a list of priors, with each list element being the prior for that escalation.
        :return:
        """
        if not isinstance(self.number_of_levels, int) or self.number_of_levels < 2:
            sys.exit('Error in create_flat_prior. Number of levels should be an integer greater than 1')
        priors = []  # Will contain each of the escalation models
        number_of_needed_escalations = self.number_of_levels - 1
        for i in range(number_of_needed_escalations):
            prior = useful_functions.create_flat_prior(self.number_of_feature_rows, scalar)
            priors.append(prior)

        self.priors = priors

    def get_number_of_feature_rows_and_columns(self):
        self.number_of_feature_rows = np.shape(self.features)[0]
        self.number_of_feature_columns = np.shape(self.features)[1]

    def get_prediction(self, desired_features=None, desired_percentile=None, random=True):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]

        p_array = np.zeros((np.shape(feature_vector)[0], self.number_of_levels - 1))
        prediction_array = np.zeros((np.shape(feature_vector)[0], self.number_of_levels))
        prediction_array[:, 0] = 1

        for model_index, model_values in enumerate(self.posteriors):
            a_vector = model_values[feature_arguments, 0]
            b_vector = model_values[feature_arguments, 1]
            if random:
                p_vector = beta.rvs(a_vector, b_vector).reshape(-1)
            else:
                if desired_percentile is None:
                    sys.exit('Error in get_prediction: Desired percentile needed if random is false')
                elif desired_percentile is 'mean':
                    p_vector = a_vector / (a_vector + b_vector)
                elif desired_percentile is 'a/b':
                    return feature_vector, a_vector, b_vector

                else:
                    p_vector = beta.ppf(desired_percentile, a_vector, b_vector)
            p_array[:, model_index] = p_vector

        for column_index, column_p in enumerate(p_array.T):  # iterating over the columns
            prediction_array[:, column_index + 1] = prediction_array[:, column_index] * column_p
            prediction_array[:, column_index] = prediction_array[:, column_index] * (1 - column_p)

        return feature_vector, prediction_array

    def get_ab(self, desired_features=None):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]

        a_array = np.zeros((np.shape(feature_vector)[0], self.number_of_levels - 1))
        b_array = np.zeros((np.shape(feature_vector)[0], self.number_of_levels - 1))

        for model_index, model_values in enumerate(self.posteriors):
            a_vector = model_values[feature_arguments, 0]
            b_vector = model_values[feature_arguments, 1]

            a_array[:, model_index] = a_vector
            b_array[:, model_index] = b_vector

        return feature_vector, a_array, b_array

    def init_data(self):
        self.data_features = np.asarray([]).reshape(-1, self.number_of_feature_columns)
        self.data_values = np.asarray([]).astype(int).reshape(-1, 1)

    def init_likelihood(self):
        """
        Creates an attribute 'likelihood'.
        :return:
        """
        if not isinstance(self.number_of_levels, int) or self.number_of_levels < 2:
            sys.exit('Error in init_likelihoods. Number of levels should be an integer greater than 1')
        likelihoods = []  # Will contain each of the escalation models
        number_of_needed_escalations = self.number_of_levels - 1
        for model in range(number_of_needed_escalations):
            likelihood = useful_functions.create_zero_prior(self.number_of_feature_rows)
            likelihoods.append(likelihood.copy())

        self.likelihoods = likelihoods

    def update_data_sizes(self):
        data_sizes = np.zeros(self.number_of_levels - 1)
        for data_index, data_value in enumerate(self.data_values):
            data_value = data_value[0]
            for model_index in useful_functions.model_ranger(data_value, self.number_of_levels):
                data_sizes[model_index] += 1
        self.data_sizes = data_sizes

    def update_likelihood_efficient(self):
        if not hasattr(self, 'new_data_values'):  # Breaks out if new data hasn;t been set up
            return
        for data_index, data_value in enumerate(self.new_data_values):  # Iterate over data rows
            data_value = data_value[0]
            for feature_row_index, feature_row in enumerate(self.features):  # Iterate over feature rows
                for model_index in useful_functions.model_ranger(data_value,
                                                                 self.number_of_levels):  # iterate over model levels
                    x = feature_row
                    x_prime = self.new_data_features[data_index]
                    kernel = self.kernel.eval(x, x_prime, None)
                    if data_value > model_index:
                        self.likelihoods[model_index][feature_row_index][0] += kernel
                    else:
                        self.likelihoods[model_index][feature_row_index][1] += kernel

    def update_likelihood_inefficient(self):
        if not hasattr(self, 'new_data_values'):  # Breaks out if new data hasn;t been set up
            return
        for index, likelihood in enumerate(self.likelihoods):  # 0 all likelihoods
            self.likelihoods[index] = likelihood * 0
        self.update_data_sizes()
        for data_index, data_value in enumerate(self.data_values):  # Iterate over data rows
            data_value = data_value[0]
            for feature_row_index, feature_row in enumerate(self.features):  # Iterate over feature rows
                for model_index in useful_functions.model_ranger(data_value,
                                                                 self.number_of_levels):  # iterate over model levels
                    x = feature_row
                    data_size = self.data_sizes[model_index]
                    x_prime = self.data_features[data_index]
                    kernel = self.kernel.eval(x, x_prime, data_size)
                    if data_value > model_index:
                        self.likelihoods[model_index][feature_row_index][0] += kernel
                    else:
                        self.likelihoods[model_index][feature_row_index][1] += kernel

    def update_likelihood(self):
        if self.efficient_kernel:
            self.update_likelihood_efficient()
        else:
            self.update_likelihood_inefficient()

    def update_posterior(self):
        posteriors = []
        number_of_needed_escalations = self.number_of_levels - 1
        for model_index in range(number_of_needed_escalations):
            prior = self.priors[model_index]
            likelihood = self.likelihoods[model_index]
            posterior = prior + likelihood
            posteriors.append(posterior)
        self.posteriors = posteriors

    def define_prior_and_setup(self, prior, position=0):
        if np.shape(self.priors[position]) == np.shape(prior):
            self.priors[position] = prior
        else:
            sys.exit('priors of wrong shape')
        self.update_likelihood()
        self.update_posterior()


class Null_Model():
    def __init__(self, feature_array, data=None):
        self.features = feature_array
        self.get_number_of_feature_rows_and_columns()
        self.init_data()
        self.init_likelihood()
        if data is not None:
            self.add_data(data)  # data is a tuple of (features, values),
            # with features being a (number of samples, number of feature columns)-nparray and
            # values being a (number of samples, 1)-nparray containing integers in [0,number of levels 1]

    def add_data(self, data):
        data_features, data_values = data
        self.data_features = np.concatenate((self.data_features, data_features))
        self.data_values = np.concatenate((self.data_values, data_values))

    def create_flat_priors(self):
        pass

    def get_number_of_feature_rows_and_columns(self):
        self.number_of_feature_rows = np.shape(self.features)[0]
        self.number_of_feature_columns = np.shape(self.features)[1]

    def get_prediction(self, desired_features=None, desired_percentile=None, random=True):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        return feature_vector, None

    def init_data(self):
        self.data_features = np.asarray([]).reshape(-1, self.number_of_feature_columns)
        self.data_values = np.asarray([]).astype(int).reshape(-1, 1)

    def init_likelihood(self):
        pass

    def update_data_sizes(self):
        pass

    def update_likelihood(self):
        pass

    def update_posterior(self):
        pass


class Latent_Quadratic_Model():
    def __init__(self, feature_array, data=None, pseudo_weight=0.01):
        self.features = feature_array
        self.pseudo_weight = pseudo_weight
        self.get_number_of_feature_rows_and_columns()
        self.init_data()
        self.init_likelihood()
        if data is not None:
            self.add_data(data)  # data is a tuple of (features, values),
            # with features being a (number of samples, number of feature columns)-nparray and
            # values being a (number of samples, 1)-nparray containing integers in [0,number of levels 1]
            self.update_likelihood()
        self.update_posterior()

    def add_data(self, data):
        data_features, data_values = data
        if (data_values >= 0).all() \
                and useful_functions.isinteger(data_values).all():
            self.data_features = np.concatenate((self.data_features, data_features))
            self.data_values = np.concatenate((self.data_values, data_values))
            self.new_data_features = data_features
            self.new_data_values = data_values
        else:
            sys.exit('Error in add_data: All values should be integers and within the number of potential levels')

    def get_number_of_feature_rows_and_columns(self):
        self.number_of_feature_rows = np.shape(self.features)[0]
        self.number_of_feature_columns = np.shape(self.features)[1]

    def get_prediction(self, desired_features=None, desired_percentile=None, random=True):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]

        prediction_array = scaled_lq(feature_vector, params=self.parameters)
        prediction_array = prediction_array.reshape(-1, 1)

        return feature_vector, prediction_array

    def init_data(self):
        self.data_features = np.asarray([]).reshape(-1, self.number_of_feature_columns)
        self.data_values = np.asarray([]).astype(int).reshape(-1, 1)

    def init_likelihood(self):
        """
        Creates an attribute 'likelihood'.
        :return:
        """
        self.pseudo_features = None
        self.pseudo_values = None

    def update_data_sizes(self):
        data_sizes = np.zeros(self.number_of_levels - 1)
        for data_index, data_value in enumerate(self.data_values):
            data_value = data_value[0]
            for model_index in useful_functions.model_ranger(data_value, self.number_of_levels):
                data_sizes[model_index] += 1
        self.data_sizes = data_sizes

    def update_likelihood(self):
        pass

    def update_posterior(self):
        self.update_parameters()

    def get_ESS(self):
        ESS = np.size(self.pseudo_values) * self.pseudo_weight
        return ESS

    def define_pseudo_data(self, psuedo_doses, pseudo_responses):
        self.pseudo_features = psuedo_doses
        self.pseudo_values = pseudo_responses

    def update_psuedo_weight(self, new_weight):
        self.pseudo_weight = new_weight
        self.update_parameters()

    def update_parameters(self):
        if self.pseudo_features is None:
            if np.size(self.data_features) == 0:
                self.parameters = np.asarray([-1, 0.6, -0.05])
                return None
            else:
                dose = self.data_features
                response = self.data_values
                weight = None
        else:
            if np.size(self.data_features) == 0:
                dose = self.pseudo_features
                response = self.pseudo_values
                weight = None
            else:
                dose = self.data_features
                response = self.data_values
                dose, response, weight = combine_data_and_pseudo(self.pseudo_features, self.pseudo_values,
                                                                 self.data_features, self.data_values,
                                                                 self.pseudo_weight)

        optimisation_results = lq_callibrate(dose, response, guess=None, weight_eff=weight)
        self.parameters = optimisation_results.x


class Saturating_Model():
    def __init__(self, feature_array, data=None, pseudo_weight=0.01):
        self.features = feature_array
        self.pseudo_weight = pseudo_weight
        self.get_number_of_feature_rows_and_columns()
        self.init_data()
        self.init_likelihood()
        if data is not None:
            self.add_data(data)  # data is a tuple of (features, values),
            # with features being a (number of samples, number of feature columns)-nparray and
            # values being a (number of samples, 1)-nparray containing integers in [0,number of levels 1]
            self.update_likelihood()
        self.update_posterior()

    def add_data(self, data):
        data_features, data_values = data
        if (data_values >= 0).all() \
                and useful_functions.isinteger(data_values).all():
            self.data_features = np.concatenate((self.data_features, data_features))
            self.data_values = np.concatenate((self.data_values, data_values))
            self.new_data_features = data_features
            self.new_data_values = data_values
        else:
            sys.exit('Error in add_data: All values should be integers and within the number of potential levels')

    def get_number_of_feature_rows_and_columns(self):
        self.number_of_feature_rows = np.shape(self.features)[0]
        self.number_of_feature_columns = np.shape(self.features)[1]

    def get_prediction(self, desired_features=None, desired_percentile=None, random=True):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]

        prediction_array = scaled_saturating(feature_vector, params=self.parameters)
        prediction_array = prediction_array.reshape(-1, 1)

        return feature_vector, prediction_array

    def init_data(self):
        self.data_features = np.asarray([]).reshape(-1, self.number_of_feature_columns)
        self.data_values = np.asarray([]).astype(int).reshape(-1, 1)

    def init_likelihood(self):
        """
        Creates an attribute 'likelihood'.
        :return:
        """
        self.pseudo_features = None
        self.pseudo_values = None

    def update_data_sizes(self):
        data_sizes = np.zeros(self.number_of_levels - 1)
        for data_index, data_value in enumerate(self.data_values):
            data_value = data_value[0]
            for model_index in useful_functions.model_ranger(data_value, self.number_of_levels):
                data_sizes[model_index] += 1
        self.data_sizes = data_sizes

    def update_likelihood(self):
        pass

    def update_posterior(self):
        self.update_parameters()

    def get_ESS(self):
        ESS = np.size(self.pseudo_values) * self.pseudo_weight
        return ESS

    def define_pseudo_data(self, psuedo_doses, pseudo_responses):
        self.pseudo_features = psuedo_doses
        self.pseudo_values = pseudo_responses

    def update_psuedo_weight(self, new_weight):
        self.pseudo_weight = new_weight
        self.update_parameters()

    def update_parameters(self):
        if self.pseudo_features is None:
            if np.size(self.data_features) == 0:
                self.parameters = np.asarray([-1, 0.6])
                return None
            else:
                dose = self.data_features
                response = self.data_values
                weight = None
        else:
            if np.size(self.data_features) == 0:
                dose = self.pseudo_features
                response = self.pseudo_values
                weight = None
            else:
                dose, response, weight = combine_data_and_pseudo(self.pseudo_features, self.pseudo_values,
                                                                 self.data_features, self.data_values,
                                                                 self.pseudo_weight)

        optimisation_results = saturating_callibrate(dose, response, guess=None, weight_eff=weight)
        self.parameters = optimisation_results.x


class Saturating_2D_Model():
    def __init__(self, feature_array, data=None, pseudo_weight=0.01):
        self.features = feature_array
        self.pseudo_weight = pseudo_weight
        self.get_number_of_feature_rows_and_columns()
        self.init_data()
        self.init_likelihood()
        if data is not None:
            self.add_data(data)  # data is a tuple of (features, values),
            # with features being a (number of samples, number of feature columns)-nparray and
            # values being a (number of samples, 1)-nparray containing integers in [0,number of levels 1]
            self.update_likelihood()
        self.update_posterior()

    def add_data(self, data):
        data_features, data_values = data
        if (data_values >= 0).all() \
                and useful_functions.isinteger(data_values).all():
            self.data_features = np.concatenate((self.data_features, data_features))
            self.data_values = np.concatenate((self.data_values, data_values))
            self.new_data_features = data_features
            self.new_data_values = data_values
        else:
            sys.exit('Error in add_data: All values should be integers and within the number of potential levels')

    def get_number_of_feature_rows_and_columns(self):
        self.number_of_feature_rows = np.shape(self.features)[0]
        self.number_of_feature_columns = np.shape(self.features)[1]

    def get_prediction(self, desired_features=None, desired_percentile=None, random=True):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]

        prediction_array = scaled_saturating_2D(feature_vector, params=self.parameters)
        prediction_array = prediction_array.reshape(-1, 1)

        return feature_vector, prediction_array

    def init_data(self):
        self.data_features = np.asarray([]).reshape(-1, self.number_of_feature_columns)
        self.data_values = np.asarray([]).astype(int).reshape(-1, 1)

    def init_likelihood(self):
        """
        Creates an attribute 'likelihood'.
        :return:
        """
        self.pseudo_features = None
        self.pseudo_values = None

    def update_data_sizes(self):
        data_sizes = np.zeros(self.number_of_levels - 1)
        for data_index, data_value in enumerate(self.data_values):
            data_value = data_value[0]
            for model_index in useful_functions.model_ranger(data_value, self.number_of_levels):
                data_sizes[model_index] += 1
        self.data_sizes = data_sizes

    def update_likelihood(self):
        pass

    def update_posterior(self):
        self.update_parameters()

    def get_ESS(self):
        ESS = np.size(self.pseudo_values) * self.pseudo_weight
        return ESS

    def define_pseudo_data(self, psuedo_doses, pseudo_responses):
        self.pseudo_features = psuedo_doses
        self.pseudo_values = pseudo_responses

    def update_psuedo_weight(self, new_weight):
        self.pseudo_weight = new_weight
        self.update_parameters()

    def update_parameters(self):
        if self.pseudo_features is None:
            if np.size(self.data_features) == 0:
                self.parameters = np.asarray([-1, 0.6])
                return None
            else:
                dose = self.data_features
                response = self.data_values
                weight = None
        else:
            if np.size(self.data_features) == 0:
                dose = self.pseudo_features
                response = self.pseudo_values
                weight = None
            else:
                dose, response, weight = combine_data_and_pseudo(self.pseudo_features, self.pseudo_values,
                                                                 self.data_features, self.data_values,
                                                                 self.pseudo_weight)

        optimisation_results = saturating_2D_callibrate(dose, response, guess=None, weight_eff=weight)
        self.parameters = optimisation_results.x


class Latent_Quadratic_Model_2D():
    def __init__(self, feature_array, data=None, pseudo_weight=0.01):
        self.features = feature_array
        self.pseudo_weight = pseudo_weight
        self.get_number_of_feature_rows_and_columns()
        self.init_data()
        self.init_likelihood()
        if data is not None:
            self.add_data(data)  # data is a tuple of (features, values),
            # with features being a (number of samples, number of feature columns)-nparray and
            # values being a (number of samples, 1)-nparray containing integers in [0,number of levels 1]
            self.update_likelihood()
        self.update_posterior()

    def add_data(self, data):
        data_features, data_values = data
        if (data_values >= 0).all() \
                and useful_functions.isinteger(data_values).all():
            self.data_features = np.concatenate((self.data_features, data_features))
            self.data_values = np.concatenate((self.data_values, data_values))
            self.new_data_features = data_features
            self.new_data_values = data_values
        else:
            sys.exit('Error in add_data: All values should be integers and within the number of potential levels')

    def get_number_of_feature_rows_and_columns(self):
        self.number_of_feature_rows = np.shape(self.features)[0]
        self.number_of_feature_columns = np.shape(self.features)[1]

    def get_prediction(self, desired_features=None, desired_percentile=None, random=True):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]

        prediction_array = scaled_lq_2D(feature_vector, params=self.parameters)
        prediction_array = prediction_array.reshape(-1, 1)
        return feature_vector, prediction_array

    def init_data(self):
        self.data_features = np.asarray([]).reshape(-1, self.number_of_feature_columns)
        self.data_values = np.asarray([]).astype(int).reshape(-1, 1)

    def init_likelihood(self):
        """
        Creates an attribute 'likelihood'.
        :return:
        """
        self.pseudo_features = None
        self.pseudo_values = None

    def update_data_sizes(self):
        data_sizes = np.zeros(self.number_of_levels - 1)
        for data_index, data_value in enumerate(self.data_values):
            data_value = data_value[0]
            for model_index in useful_functions.model_ranger(data_value, self.number_of_levels):
                data_sizes[model_index] += 1
        self.data_sizes = data_sizes

    def update_likelihood(self):
        pass

    def update_posterior(self):
        self.update_parameters()

    def get_ESS(self):
        ESS = np.size(self.pseudo_values) * self.pseudo_weight
        return ESS

    def define_pseudo_data(self, psuedo_doses, pseudo_responses):
        self.pseudo_features = psuedo_doses
        self.pseudo_values = pseudo_responses

    def update_psuedo_weight(self, new_weight):
        self.pseudo_weight = new_weight
        self.update_parameters()

    def update_parameters(self):
        if self.pseudo_features is None:
            if np.size(self.data_features) == 0:
                self.parameters = np.asarray([-2, 0.6, -0.05, 0.6, -0.05])
                return None
            else:
                dose = self.data_features
                response = self.data_values
                weight = None
        else:
            if np.size(self.data_features) == 0:
                dose = self.pseudo_features
                response = self.pseudo_values
                weight = None
            else:
                dose = self.data_features
                response = self.data_values
                dose, response, weight = combine_data_and_pseudo(self.pseudo_features, self.pseudo_values,
                                                                 self.data_features, self.data_values,
                                                                 self.pseudo_weight)
        optimisation_results = lq_callibrate_2D(dose, response, guess=None, weight_eff=weight)
        self.parameters = optimisation_results.x

    def plot_prediction(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for _ in range(5):
            feature_vector, prediction_array = self.get_prediction()
            x = feature_vector[:, 0].reshape(-1)
            y = feature_vector[:, 1].reshape(-1)
            z = prediction_array
            ax.scatter3D(x, y, z)
        if np.size(self.data_features) > 0:
            ax.scatter3D(self.data_features[:, 0], self.data_features[:, 1], self.data_values)
        ax.view_init(elev=45., azim=225)
        plt.show()


class Latent_Quadratic_Model_3D():
    def __init__(self, feature_array, data=None, pseudo_weight=0.01):
        self.features = feature_array
        self.pseudo_weight = pseudo_weight
        self.get_number_of_feature_rows_and_columns()
        self.init_data()
        self.init_likelihood()
        if data is not None:
            self.add_data(data)  # data is a tuple of (features, values),
            # with features being a (number of samples, number of feature columns)-nparray and
            # values being a (number of samples, 1)-nparray containing integers in [0,number of levels 1]
            self.update_likelihood()
        self.update_posterior()

    def add_data(self, data):
        data_features, data_values = data
        if (data_values >= 0).all() \
                and useful_functions.isinteger(data_values).all():
            self.data_features = np.concatenate((self.data_features, data_features))
            self.data_values = np.concatenate((self.data_values, data_values))
            self.new_data_features = data_features
            self.new_data_values = data_values
        else:
            sys.exit('Error in add_data: All values should be integers and within the number of potential levels')

    def get_number_of_feature_rows_and_columns(self):
        self.number_of_feature_rows = np.shape(self.features)[0]
        self.number_of_feature_columns = np.shape(self.features)[1]

    def get_prediction(self, desired_features=None, desired_percentile=None, random=True):
        if desired_features is None:
            feature_arguments = []
            for d in self.features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]
        else:
            feature_arguments = []
            for d in desired_features:
                argument = np.where((self.features == d).all(axis=1))[0]
                feature_arguments.append(argument.copy())
            feature_arguments = np.asarray(feature_arguments).reshape(-1)
            if np.size(feature_arguments) == 0:
                sys.exit('Error in get_prediction: No such feature found in the model')
            feature_vector = self.features[feature_arguments]

        prediction_array = scaled_lq_3D(feature_vector, params=self.parameters)
        prediction_array = prediction_array.reshape(-1, 1)
        return feature_vector, prediction_array

    def init_data(self):
        self.data_features = np.asarray([]).reshape(-1, self.number_of_feature_columns)
        self.data_values = np.asarray([]).astype(int).reshape(-1, 1)

    def init_likelihood(self):
        """
        Creates an attribute 'likelihood'.
        :return:
        """
        self.pseudo_features = None
        self.pseudo_values = None

    def update_data_sizes(self):
        data_sizes = np.zeros(self.number_of_levels - 1)
        for data_index, data_value in enumerate(self.data_values):
            data_value = data_value[0]
            for model_index in useful_functions.model_ranger(data_value, self.number_of_levels):
                data_sizes[model_index] += 1
        self.data_sizes = data_sizes

    def update_likelihood(self):
        pass

    def update_posterior(self):
        self.update_parameters()

    def get_ESS(self):
        ESS = np.size(self.pseudo_values) * self.pseudo_weight
        return ESS

    def define_pseudo_data(self, psuedo_doses, pseudo_responses):
        self.pseudo_features = psuedo_doses
        self.pseudo_values = pseudo_responses

    def update_psuedo_weight(self, new_weight):
        self.pseudo_weight = new_weight
        self.update_parameters()

    def update_parameters(self):
        if self.pseudo_features is None:
            if np.size(self.data_features) == 0:
                self.parameters = np.asarray([-2, 0.6, -0.05, 0.6, -0.05])
                return None
            else:
                dose = self.data_features
                response = self.data_values
                weight = None
        else:
            if np.size(self.data_features) == 0:
                dose = self.pseudo_features
                response = self.pseudo_values
                weight = None
            else:
                dose, response, weight = combine_data_and_pseudo(self.pseudo_features, self.pseudo_values,
                                                                 self.data_features, self.data_values,
                                                                 self.pseudo_weight)
        optimisation_results = lq_callibrate_3D(dose, response, guess=None, weight_eff=weight)
        self.parameters = optimisation_results.x
