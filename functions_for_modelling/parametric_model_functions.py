import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')


def scale_lq_params(params):
    mu, b1, b2 = params
    mu, b1, b2 = mu, min(60, abs(b1)), -abs(
        b2)
    return_array = np.array([mu, b1, b2])
    return return_array


def scaled_lq(independant, params):
    mu, b1, b2 = scale_lq_params(params)
    alpha = mu + b1 * independant + b2 * independant ** 2
    p = np.exp(- alpha)
    p = 1 / (1 + p)
    dependant = p
    return dependant


def likelihood_lq(params, doses, results, weight_eff):
    probs = scaled_lq(doses, params)
    yhat = probs
    likelihood_array = yhat * results + (1 - yhat) * (1 - results)
    likelihood_array = likelihood_array.reshape(-1)
    likelihood_array = np.log(likelihood_array)
    likelihood_array = weight_eff * likelihood_array
    negLL = -np.sum(likelihood_array)
    return (negLL)


def lq_callibrate(eff_x, eff_y, guess=None, weight_eff=None):
    if guess is None:
        guess = np.array([-1, 0.6, -0.05])  # arbritary guess, but appears reasonable
    if weight_eff is None:
        length = np.size(eff_x)
        weight_eff = np.ones(length)
    results = minimize(likelihood_lq, guess, args=(eff_x, eff_y, weight_eff))
    return results


def combine_data_and_pseudo(pseudo_x, pseudo_y, actual_x, actual_y, weight_pseudo=0.01):
    x = np.concatenate((pseudo_x, actual_x), axis=0)
    y = np.concatenate((pseudo_y, actual_y), axis=0).astype(int)
    weight_pseudo = np.full((len(pseudo_x)), weight_pseudo)
    weight_actual = np.full((len(actual_x)), 1)
    weight = np.concatenate((weight_pseudo, weight_actual), axis=0)
    return x, y, weight


def scale_lq_params_2D(params):
    mu, b1, b2, c1, c2 = params
    mu, b1, b2, c1, c2 = mu, min(60, abs(b1)), -abs(b2), min(60, abs(c1)), -abs(
        c2)
    return_array = np.array([mu, b1, b2, c1, c2])
    return return_array


def scaled_lq_2D(independant, params):
    x, y = independant[:, 0], independant[:, 1]
    mu, b1, b2, c1, c2 = scale_lq_params_2D(params)
    alpha = mu + b1 * x + b2 * x ** 2 + c1 * y + c2 * y ** 2
    p = np.exp(- alpha)
    p = 1 / (1 + p)
    dependant = p
    return dependant


def likelihood_lq_2D(params, doses, results, weight_eff):
    probs = scaled_lq_2D(doses, params)
    yhat = probs.reshape(-1, 1)
    likelihood_array = yhat * results + (1 - yhat) * (1 - results)
    likelihood_array = likelihood_array.reshape(-1)
    likelihood_array = np.log(likelihood_array)
    likelihood_array = weight_eff * likelihood_array
    negLL = -np.sum(likelihood_array)
    return (negLL)


def lq_callibrate_2D(eff_x, eff_y, guess=None, weight_eff=None):
    if guess is None:
        guess = np.array([-2, 0.6, -0.05, 0.6, -0.05])  # arbritary guess, but appears reasonable
    if weight_eff is None:
        length = np.shape(eff_x)[0]
        weight_eff = np.ones(length)
    results = minimize(likelihood_lq_2D, guess, args=(eff_x, eff_y, weight_eff))
    return results


def scale_lq_params_3D(params):
    mu, b1, b2, c1, c2, d1, d2 = params
    mu, b1, b2, c1, c2, d1, d2 = mu, min(60, abs(b1)), -abs(b2), min(60, abs(c1)), -abs(c2), min(60, abs(d1)), -abs(
        d2)
    return_array = np.array([mu, b1, b2, c1, c2, d1, d2])
    return return_array


def scaled_lq_3D(independant, params):
    x, y, z = independant[:, 0], independant[:, 1], independant[:, 2]
    mu, b1, b2, c1, c2, d1, d2 = scale_lq_params_3D(params)
    alpha = mu + b1 * x + b2 * x ** 2 + c1 * y + c2 * y ** 2 + d1 * z + d2 * z ** 2
    p = np.exp(- alpha)
    p = 1 / (1 + p)
    dependant = p
    return dependant


def likelihood_lq_3D(params, doses, results, weight_eff):
    probs = scaled_lq_3D(doses, params)
    yhat = probs.reshape(-1, 1)
    likelihood_array = yhat * results + (1 - yhat) * (1 - results)
    likelihood_array = likelihood_array.reshape(-1)
    likelihood_array = np.log(likelihood_array)
    likelihood_array = weight_eff * likelihood_array
    negLL = -np.sum(likelihood_array)
    return negLL


def lq_callibrate_3D(eff_x, eff_y, guess=None, weight_eff=None):
    if guess is None:
        guess = np.array([-2, 0.6, -0.05, 0.6, -0.05, 0.6, -0.05])  # arbritary guess, but appears reasonable
    if weight_eff is None:
        length = np.shape(eff_x)[0]
        weight_eff = np.ones(length)
    results = minimize(likelihood_lq_3D, guess, args=(eff_x, eff_y, weight_eff))
    return results


def scale_saturating_params(params):
    mu, b1 = params
    mu, b1 = mu, min(60,
                     abs(b1))
    return_array = np.array([mu, b1])
    return return_array


def scaled_saturating(independant, params):
    mu, b1 = scale_saturating_params(params)
    alpha = mu + b1 * independant
    p = np.exp(- alpha)
    p = 1 / (1 + p)
    dependant = p
    return dependant


def likelihood_saturating(params, doses, results, weight_eff):
    probs = scaled_saturating(doses, params)
    yhat = probs.reshape(-1, 1)
    likelihood_array = yhat * results + (1 - yhat) * (1 - results)
    likelihood_array = likelihood_array.reshape(-1)
    likelihood_array = np.log(likelihood_array)
    likelihood_array = weight_eff * likelihood_array
    negLL = -np.sum(likelihood_array)
    return negLL


def saturating_callibrate(eff_x, eff_y, guess=None, weight_eff=None):
    if guess is None:
        guess = np.array([-2, 0.6])  # arbritary guess, but appears reasonable
    if weight_eff is None:
        length = np.shape(eff_x)[0]
        weight_eff = np.ones(length)
    results = minimize(likelihood_saturating, guess, args=(eff_x, eff_y, weight_eff))
    return results


def scale_saturating_2D_params(params):
    mu, b1, b2 = params
    mu, b1, b2 = mu, min(60, abs(b1)), min(60, abs(
        b2))
    return_array = np.array([mu, b1, b2])
    return return_array


def scaled_saturating_2D(independant, params):  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2931331/
    x, y = independant[:, 0], independant[:, 1]
    mu, b1, b2 = scale_saturating_2D_params(params)
    alpha = mu + b1 * x + b2 * y
    p = np.exp(- alpha)
    p = 1 / (1 + p)
    dependant = p
    return dependant


def likelihood_saturating_2D(params, doses, results, weight_eff):
    probs = scaled_saturating_2D(doses, params)
    yhat = probs.reshape(-1, 1)
    likelihood_array = yhat * results + (1 - yhat) * (1 - results)
    likelihood_array = likelihood_array.reshape(-1)
    likelihood_array = np.log(likelihood_array)
    likelihood_array = weight_eff * likelihood_array
    negLL = -np.sum(likelihood_array)
    return negLL


def saturating_2D_callibrate(eff_x, eff_y, guess=None, weight_eff=None):
    if guess is None:
        guess = np.array([-2, 0.6, 0.6])  # arbritary guess, but appears reasonable
    if weight_eff is None:
        length = np.shape(eff_x)[0]
        weight_eff = np.ones(length)
    results = minimize(likelihood_saturating_2D, guess, args=(eff_x, eff_y, weight_eff))
    return results
