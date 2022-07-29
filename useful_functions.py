import numpy as np
import sys


def combine_possible_features(*feature_vectors):
    count_of_features = 0
    if isinstance(feature_vectors[0], np.ndarray):  # logic for if there are multiple feature vectors
        for _ in feature_vectors:
            count_of_features += 1
        mesh = np.meshgrid(*feature_vectors)
        mesh = np.transpose(mesh)
        mesh = mesh.reshape(-1, count_of_features)
    elif isinstance(feature_vectors[0], np.int64):
        mesh = np.transpose(feature_vectors).reshape(-1, 1)
    else:
        sys.exit('Error: Unknown format for *feature_vectors in combine_possible_features.')
    return mesh


def create_flat_prior(number_of_feature_rows, scalar=None):
    if not isinstance(number_of_feature_rows, int) or number_of_feature_rows < 1:
        sys.exit('Error in create_flat_prior. Number of feature_rows should be an integer greater than 0')
    a_vec = np.ones(number_of_feature_rows)
    b_vec = np.ones(number_of_feature_rows)
    if not scalar is None:
        a_vec *= scalar
        b_vec *= scalar
    prior = np.column_stack((a_vec, b_vec))
    return prior


def create_zero_prior(number_of_feature_rows):
    if not isinstance(number_of_feature_rows, int) or number_of_feature_rows < 1:
        sys.exit('Error in create_flat_prior. Number of feature_rows should be an integer greater than 0')
    a_vec = np.zeros(number_of_feature_rows)
    b_vec = np.zeros(number_of_feature_rows)
    prior = np.column_stack((a_vec, b_vec))
    return prior


def isinteger(x):
    return np.equal(np.mod(x, 1), 0)


def model_ranger(x, total_number_of_models):  # returns the range of models that need effecting for a given value
    x2 = min(x + 1, total_number_of_models - 1)
    return list(range(x2))


def randargmax(vector):
    """
    Returns a random argument of the vector which is one of the maximum possible values of that vector.
    :param vector:
    :return:
    """
    return np.argmax(np.random.random(vector.shape) * (vector == vector.max()))


def randargmin(vector):
    """
     Returns a random argument of the vector which is one of the minimum possible values of that vector.
    :param vector:
    :return:
    """
    return np.argmax(np.random.random(vector.shape) * (vector == vector.min()))


def sample_from_probability(prob_vec):
    vec_length = len(prob_vec)
    randarray = np.random.rand(vec_length)
    sample = prob_vec > randarray
    sample = sample.astype(int)
    return sample


def sample_from_ordinal_probability(prob_array):
    shape = np.shape(prob_array)
    num_samples = shape[0]
    cum_array = np.cumsum(prob_array, axis=1)
    randarray = np.random.rand(num_samples)
    sample = []
    for index, column in enumerate(cum_array):
        value = np.argmax(column >= randarray[index])
        sample.append(value)
    return sample


# Uses scipy minimise to find parameters (a,b) such that a beta distribution defined by those parameter has the desired 95% confidence interval.

def beta_mode_to_parameters(expected, confidence):
    """
    :param expected: the expected p for the given dose
    :param confidence: how many individuals worth of confidence to would like to express this as
    :param initial_guess:
    :return:
    """
    a = expected * confidence + 1
    b = (1 - expected) * confidence + 1
    return np.asarray([a, b]).reshape((2))


def beta_mode_array_to_parameters(expected_vector, confidence_vector):
    """
    takes an array of believed confidence intervals for the true probability and returns appropriate beta parameters.
    used for establishing priors
    :param lower_ci_vector:
    :param upper_ci_vector:
    :return:
    """
    if np.shape(expected_vector) != np.shape(confidence_vector):
        print('Error: CI vectors are not or same size')
        return None
    return_array = np.zeros((np.size(confidence_vector), 2))
    for index, _ in enumerate(expected_vector):
        callibrated_parameters = beta_mode_to_parameters(expected_vector[index],
                                                         confidence_vector[index])
        return_array[index, :] = callibrated_parameters
    return return_array


def soft_max(utilities, inverse_temperature):  # Calculates probability of selecting based on utility
    # A small inverse temperature has more exploration, a large inverse temperature has high exploitation
    scaling = inverse_temperature * utilities

    max_scaling = np.amax(scaling)  ## This section is to avoid issues with overflow
    if max_scaling >= 705:
        subtraction = max_scaling - 705
        scaling = scaling - subtraction

    utility_score = np.exp(scaling)
    total_score = np.sum(utility_score)
    np.nan_to_num(total_score, copy=False)
    scaled_utilities = utility_score / total_score
    return scaled_utilities


def soft_max_selector(utilities, inverse_temperature, samples=None):
    if samples is None:
        scaled_utilities = soft_max(utilities, inverse_temperature)
        number_of_doses = np.size(scaled_utilities)
        dose_index = np.arange(number_of_doses)
        chosen_index = np.random.choice(dose_index, p=scaled_utilities)
        chosen_index = chosen_index.astype('int')
        return chosen_index
    else:
        chosen_indexes = np.empty(samples)
        for i in range(samples):
            scaled_utilities = soft_max(utilities, inverse_temperature)
            number_of_doses = np.size(scaled_utilities)
            dose_index = np.arange(number_of_doses)
            chosen_indexes[i] = np.random.choice(dose_index, p=scaled_utilities)

        chosen_indexes = chosen_indexes.astype('int')
        return chosen_indexes


def generate_pseudodata(doses, efficacy_at_dose, no_efficacy_at_dose):
    pseudo_x = np.array([])
    pseudo_y = np.array([])
    for index, dose in enumerate(doses):
        total = efficacy_at_dose[index] + no_efficacy_at_dose[index]
        x_for_this_dose = np.full(total, dose)
        eff_for_this_dose = np.full(efficacy_at_dose[index], 1)
        no_eff_for_this_dose = np.full(no_efficacy_at_dose[index], 0)
        pseudo_x = np.concatenate((pseudo_x, x_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, eff_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, no_eff_for_this_dose), axis=0)

    pseudo_x = pseudo_x.reshape(-1, 1)
    pseudo_y = pseudo_y.reshape((-1, 1))
    return pseudo_x, pseudo_y


def generate_pseudodata_2D(doses, efficacy_at_dose, no_efficacy_at_dose):
    pseudo_x = np.array([]).reshape(-1, 2)
    pseudo_y = np.array([])
    for index, dose in enumerate(doses):
        total = efficacy_at_dose[index] + no_efficacy_at_dose[index]
        x_for_this_dose = np.full((total, 2), dose)
        eff_for_this_dose = np.full(efficacy_at_dose[index], 1)
        no_eff_for_this_dose = np.full(no_efficacy_at_dose[index], 0)
        pseudo_x = np.concatenate((pseudo_x, x_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, eff_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, no_eff_for_this_dose), axis=0)

    pseudo_x = pseudo_x.reshape(-1, 2)
    pseudo_y = pseudo_y.reshape((-1, 1))
    return pseudo_x, pseudo_y


def generate_pseudodata_3D(doses, efficacy_at_dose, no_efficacy_at_dose):
    pseudo_x = np.array([]).reshape(-1, 3)
    pseudo_y = np.array([])
    for index, dose in enumerate(doses):
        total = efficacy_at_dose[index] + no_efficacy_at_dose[index]
        x_for_this_dose = np.full((total, 3), dose)
        eff_for_this_dose = np.full(efficacy_at_dose[index], 1)
        no_eff_for_this_dose = np.full(no_efficacy_at_dose[index], 0)
        pseudo_x = np.concatenate((pseudo_x, x_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, eff_for_this_dose), axis=0)
        pseudo_y = np.concatenate((pseudo_y, no_eff_for_this_dose), axis=0)

    pseudo_x = pseudo_x.reshape(-1, 3)
    pseudo_y = pseudo_y.reshape((-1, 1))
    return pseudo_x, pseudo_y


"""
Functions for environment generation
"""


def distance_calcuator_1D(p1, p2):
    d0 = p1[0] - p2[0]
    d = np.sqrt(d0 ** 2)
    return d


def distance_calcuator_2D(p1, p2):
    d0 = p1[0] - p2[0]
    d1 = p1[1] - p2[1]
    d = np.sqrt(d0 ** 2 + d1 ** 2)
    return d


def distance_calcuator_3D(p1, p2):
    d0 = p1[0] - p2[0]
    d1 = p1[1] - p2[1]
    d2 = p1[2] - p2[2]
    d = np.sqrt(d0 ** 2 + d1 ** 2 + d2 ** 2)
    return d


def _smooth_1D(data, value, m=2, xlim=(0, 1), xres=6):
    xs = np.linspace(xlim[0], xlim[1], xres)
    points = []
    values = []
    for x in xs:
        points.append([x])
        values.append([0])
    points = np.asarray(points)
    values = np.asarray(values).astype('float32')

    for indexp, point in enumerate(points):
        distances = np.ones((m)) * np.inf
        vals = np.ones((m))
        for indexdat, datum in enumerate(data):
            d_star = distance_calcuator_1D(datum, point)
            for indexdis, distance in enumerate(distances):
                if d_star < distance:
                    distances[indexdis] = d_star
                    vals[indexdis] = value[indexdat][0]
                    break
            argo = (-distances).argsort()
            distances = distances[argo]
            vals = vals[argo]
        mean = np.mean(vals)
        values[indexp, 0] = mean

    return points, values


def smooth_1D(data, value, iterations=10, m=2, xlim=(0, 1), xres=6):
    p, v = _smooth_1D(data, value, 1, xlim, xres)
    for i in range(iterations):
        print(i)
        p = np.row_stack((data, p))
        v = np.row_stack((value, v))
        p, v = _smooth_1D(p, v, m, xlim, xres)
    p, v = _smooth_1D(p, v, m, xlim, xres)
    return p, v


def _smooth_2D(data, value, m=2, xlim=(0, 1), ylim=(0, 1), xres=6, yres=6):
    xs = np.linspace(xlim[0], xlim[1], xres)
    ys = np.linspace(ylim[0], ylim[1], yres)
    points = []
    values = []
    for x in xs:
        for y in ys:
            points.append([x, y])
            values.append([0])
    points = np.asarray(points)
    values = np.asarray(values).astype('float32')

    for indexp, point in enumerate(points):
        distances = np.ones((m)) * np.inf
        vals = np.ones((m))
        for indexdat, datum in enumerate(data):
            d_star = distance_calcuator_2D(datum, point)
            for indexdis, distance in enumerate(distances):
                if d_star < distance:
                    distances[indexdis] = d_star
                    vals[indexdis] = value[indexdat][0]
                    break
            argo = (-distances).argsort()
            distances = distances[argo]
            vals = vals[argo]
        mean = np.mean(vals)
        values[indexp, 0] = mean

    return points, values


def smooth_2D(data, value, iterations=10, m=2, xlim=(0, 1), ylim=(0, 1), xres=6, yres=6):
    p, v = _smooth_2D(data, value, 1, xlim, ylim, xres, yres)
    for i in range(iterations):
        print(i)
        p = np.row_stack((data, p))
        v = np.row_stack((value, v))
        p, v = _smooth_2D(p, v, m, xlim, ylim, xres, yres)
    return p, v


def _smooth_3D(data, value, m=2, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xres=6, yres=6, zres=6):
    xs = np.linspace(xlim[0], xlim[1], xres)
    ys = np.linspace(ylim[0], ylim[1], yres)
    zs = np.linspace(zlim[0], zlim[1], zres)
    points = []
    values = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append([x, y, z])
                values.append([0])
    points = np.asarray(points)
    values = np.asarray(values).astype('float32')

    for indexp, point in enumerate(points):
        distances = np.ones((m)) * np.inf
        vals = np.ones((m))
        for indexdat, datum in enumerate(data):
            d_star = distance_calcuator_3D(datum, point)
            for indexdis, distance in enumerate(distances):
                if d_star < distance:
                    distances[indexdis] = d_star
                    vals[indexdis] = value[indexdat][0]
                    break
            argo = (-distances).argsort()
            distances = distances[argo]
            vals = vals[argo]
        mean = np.mean(vals)
        values[indexp, 0] = mean

    return points, values


def smooth_3D(data, value, iterations=10, m=2, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xres=6, yres=6, zres=6):
    p, v = _smooth_3D(data, value, 1, xlim, ylim, zlim, xres, yres, zres)
    for i in range(iterations):
        print(i)
        p = np.row_stack((data, p))
        v = np.row_stack((value, v))
        p, v = _smooth_3D(p, v, m, xlim, ylim, zlim, xres, yres, zres)
    return p, v
  
