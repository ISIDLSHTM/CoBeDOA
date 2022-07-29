import numpy as np
import sys
import useful_functions as uf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class Experiment():
    def __init__(self, Environment,
                 Efficacy_Model,
                 Toxicity_Model,
                 Utility_Function,
                 Selection_Method,
                 Final_Selection_Method,
                 Block_Size=6):
        self.environment = Environment
        self.efficacy_model = Efficacy_Model
        self.toxicity_model = Toxicity_Model
        self.utility_function = Utility_Function
        self.selection_method = Selection_Method
        self.final_selection_method = Final_Selection_Method
        self.block_size = Block_Size

        self.initialise_arrays()
        self.sanity_check()
        self.get_an_actual_optimal_dose()

    def initialise_arrays(self):
        self.dose_data = []
        self.efficacy_data = []
        self.toxicity_data = []
        self.predicted_optimal_doses = []
        self.predicted_optimal_utilities = []
        self.actual_utility_at_predicted_optimal = []
        self.individual_utilities = []
        self.individual_regrets = []
        self.cumulative_regrets = []
        self.average_regrets = []
        self.new_doses = []

    def sanity_check(self):
        # print(self.environment.query_doses)
        # print(self.efficacy_model.features)
        # print(self.toxicity_model.features)
        pass

    def choose_next_doses(self):
        next_doses = self.selection_method.get_next_doses(self.environment, self.efficacy_model, self.toxicity_model,
                                                          self.utility_function, self.block_size)

        self.new_doses.extend(next_doses)

    def suggest_optimal_doses(self):
        actual_efficacy = self.environment.efficacy_probabilities
        actual_toxicity = self.environment.toxicity_probabilities

        best_dose, best_utility, doses, best_arg = self.final_selection_method.predict_optimal(self.environment,
                                                                                               self.efficacy_model,
                                                                                               self.toxicity_model,
                                                                                               self.utility_function,
                                                                                               self.block_size)
        actual_utilities = self.utility_function.get_dose_utility(actual_efficacy, actual_toxicity)
        actual_utility = actual_utilities[best_arg]

        for i in range(self.block_size):
            self.predicted_optimal_doses.append(list(best_dose))
            self.predicted_optimal_utilities.append(best_utility)
            self.actual_utility_at_predicted_optimal.append(actual_utility)

        return best_dose, best_utility

    def get_efficacy_data(self):
        new_efficacy_data = self.environment.sample_dose_efficacy(self.new_doses)
        self.new_efficacy_data = np.asarray(new_efficacy_data)

    def get_toxicity_data(self):
        new_toxicity_data = self.environment.sample_dose_toxicity(self.new_doses)
        self.new_toxicity_data = np.asarray(new_toxicity_data)

    def update_efficacy_model(self):
        data = self.new_doses, self.new_efficacy_data.reshape(-1, 1)
        self.efficacy_model.add_data(data)
        self.efficacy_model.update_likelihood()
        self.efficacy_model.update_posterior()

    def update_toxicity_model(self):
        data = self.new_doses, self.new_toxicity_data.reshape(-1, 1)
        self.toxicity_model.add_data(data)
        self.toxicity_model.update_likelihood()
        self.toxicity_model.update_posterior()

    def update_utility_and_regret(self):
        for individual in zip(self.new_efficacy_data, self.new_toxicity_data):
            individual_utility = self.utility_function.get_individual_utility(individual[0], individual[1])
            self.individual_utilities.append(individual_utility.copy())
            individual_regret = self.optimal_utility - individual_utility
            self.individual_regrets.append(individual_regret.copy())

        self.cumulative_regrets = np.cumsum(self.individual_regrets)
        number_of_individuals = len(self.cumulative_regrets)
        self.average_regrets = self.cumulative_regrets / (range(1, number_of_individuals + 1))

    def update_and_clear_data(self):
        self.efficacy_data.extend(self.new_efficacy_data)
        self.toxicity_data.extend(self.new_toxicity_data)
        for dose in self.new_doses:
            self.dose_data.append(dose)
        self.new_efficacy_data = []
        self.new_toxicity_data = []
        self.new_doses = []

    def get_an_actual_optimal_dose(self):
        actual_efficacy = self.environment.efficacy_probabilities
        actual_toxicity = self.environment.toxicity_probabilities
        actual_utilities = self.utility_function.get_dose_utility(actual_efficacy, actual_toxicity)
        bestarg = uf.randargmax(actual_utilities)
        self.optimal_dose = self.environment.query_doses[bestarg]
        self.optimal_utility = actual_utilities[bestarg]

    def one_loop(self):
        self.choose_next_doses()
        self.get_efficacy_data()
        self.get_toxicity_data()
        self.update_efficacy_model()
        self.update_toxicity_model()
        self.update_utility_and_regret()
        self.update_and_clear_data()
        self.suggest_optimal_doses()

    def full_loop(self, iterations):
        for i in range(iterations):
            self.choose_next_doses()
            self.get_efficacy_data()
            self.get_toxicity_data()
            self.update_efficacy_model()
            self.update_toxicity_model()
            self.update_utility_and_regret()
            self.update_and_clear_data()
            self.suggest_optimal_doses()

    def full_loop_visualise(self, iterations):
        for i in range(iterations):
            f, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
            best_predicted, response_predicted = self.suggest_optimal_doses()

            f, mid_v = self.efficacy_model.get_prediction(desired_percentile=0.5, random=False)
            _, upper_v = self.efficacy_model.get_prediction(desired_percentile=0.975, random=False)
            _, lower_v = self.efficacy_model.get_prediction(desired_percentile=0.025, random=False)
            mid_v, upper_v, lower_v = mid_v[:, 1], upper_v[:, 1], lower_v[:, 1]
            ax1.plot(f, mid_v, c='#1f77b4', label='Prediction w/ 95%')
            ax1.plot(f, upper_v, c='#1f77b4')
            ax1.plot(f, lower_v, c='#1f77b4')
            ax1.scatter(self.efficacy_model.data_features, self.efficacy_model.data_values,
                        color='black', marker='x', s=50, label='Observed Data')
            ax1.plot(f, self.environment.efficacy_probabilities, color='black', ls='--', label='True Efficacy')
            plt.ylim(-0.05, 1.05)
            ax1.scatter(best_predicted, response_predicted, label='Predicted Best')

            plt.title(f'1D Efficacy Maximisation, \n'
                      f' {i} cohorts and {self.block_size * (i)} individuals so far')
            plt.xlabel('Dose')
            plt.ylabel('Efficacy %')
            plt.legend(loc=2)

            plt.show()

            self.choose_next_doses()
            self.get_efficacy_data()
            self.get_toxicity_data()
            self.update_efficacy_model()
            self.update_toxicity_model()
            self.update_utility_and_regret()
            self.update_and_clear_data()
            self.suggest_optimal_doses()

    def full_loop_visualise_Thompson(self, iterations, thompson_iterations=10):
        for i in range(iterations):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
            best_predicted, response_predicted = self.suggest_optimal_doses()

            f, mid_v = self.efficacy_model.get_prediction(desired_percentile=0.5, random=False)
            _, upper_v = self.efficacy_model.get_prediction(desired_percentile=0.975, random=False)
            _, lower_v = self.efficacy_model.get_prediction(desired_percentile=0.025, random=False)
            mid_v, upper_v, lower_v = mid_v[:, 1], upper_v[:, 1], lower_v[:, 1]
            ax1.plot(f, mid_v, c='#1f77b4', label='Prediction w/ 95%')
            ax1.plot(f, upper_v, c='#1f77b4')
            ax1.plot(f, lower_v, c='#1f77b4')
            ax1.scatter(self.efficacy_model.data_features, self.efficacy_model.data_values,
                        color='black', marker='x', s=50, label='Observed Data')
            ax1.plot(f, self.environment.efficacy_probabilities, color='black', ls='--', label='True Efficacy')
            ax1.set_ylim(-0.05, 1.05)
            ax1.scatter(best_predicted, response_predicted, label='Predicted Best')

            ax1.set_title(f'1D Efficacy Maximisation, \n'
                          f' {i} cohorts and {self.block_size * (i)} individuals so far')
            ax1.set_xlabel('Dose')
            ax1.set_ylabel('Efficacy %')
            ax1.legend(loc=2)

            store = np.zeros_like(mid_v)
            for i in range(thompson_iterations):
                _, mid_v = self.efficacy_model.get_prediction()
                store[uf.randargmax(mid_v[:, 1])] += 1
            box = np.ones(3) / 3
            y_smooth = np.convolve(store / thompson_iterations, box, mode='same')
            ax2.scatter(f, y_smooth)
            ax2.set_ylim(0)

            ax2.set_xlabel('Dose')
            ax2.set_ylabel('Probability dose is optimal %')

            plt.show()

            self.choose_next_doses()
            self.get_efficacy_data()
            self.get_toxicity_data()
            self.update_efficacy_model()
            self.update_toxicity_model()
            self.update_utility_and_regret()
            self.update_and_clear_data()
            self.suggest_optimal_doses()

    def full_loop_visualise_2d(self, iterations):
        for i in range(iterations):
            f, mid_v = self.efficacy_model.get_prediction(desired_percentile=0.5, random=False)

            fig = plt.figure(figsize=(15, 15))
            x = f[:, 0].reshape(-1)
            y = f[:, 1].reshape(-1)
            z_p = mid_v[:, 1].reshape(-1)
            z_act = self.environment.efficacy_probabilities.reshape(-1)
            ax = plt.axes(projection='3d')
            # ax.scatter3D(x, y, z_p, label = 'Predicted')
            # ax.scatter3D(x, y, z_act , c = 'black', label = 'Actual')
            ax.plot_wireframe(x.reshape((21, 21)), y.reshape((21, 21)), z_p.reshape((21, 21)), label='Predicted',
                              ccount=11, rcount=11)
            ax.plot_wireframe(x.reshape((21, 21)), y.reshape((21, 21)), z_act.reshape((21, 21)), colors='black',
                              label='Actual', ccount=11, rcount=11)

            pred_argmax = np.argmax(z_p)
            act_argmax = np.argmax(z_act)
            ax.scatter3D(x[pred_argmax], y[pred_argmax], z_p[pred_argmax], color='blue', s=120)
            ax.scatter3D(x[act_argmax], y[act_argmax], z_act[act_argmax], color='black', s=120)
            ax.view_init(elev=45., azim=225)

            plt.title(f'2D Efficacy Maximisation, \n'
                      f' {i} cohorts and {self.block_size * (i)} individuals so far')
            plt.xlabel('Prime Dose')
            plt.ylabel('Boost Dose')
            ax.set_zlabel('Efficacy %')
            plt.legend(loc=2)

            plt.show()

            self.choose_next_doses()
            self.get_efficacy_data()
            self.get_toxicity_data()
            self.update_efficacy_model()
            self.update_toxicity_model()
            self.update_utility_and_regret()
            self.update_and_clear_data()
            self.suggest_optimal_doses()

    def full_loop_visualise_utility(self, iterations, exact=False):
        for i in range(iterations):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))

            f, mid_v = self.efficacy_model.get_prediction(desired_percentile=0.5, random=False)
            _, upper_v = self.efficacy_model.get_prediction(desired_percentile=0.975, random=False)
            _, lower_v = self.efficacy_model.get_prediction(desired_percentile=0.025, random=False)
            mid_v, upper_v, lower_v = mid_v[:, 1], upper_v[:, 1], lower_v[:, 1]
            ax1.plot(f, mid_v, c='#1f77b4')
            ax1.plot(f, upper_v, c='#1f77b4')
            ax1.plot(f, lower_v, c='#1f77b4')
            ax1.scatter(self.efficacy_model.data_features, self.efficacy_model.data_values,
                        color='black', marker='x', s=50)
            ax1.plot(f, self.environment.efficacy_probabilities, color='black', ls='--')
            ax1.set_ylim(-0.05, 1.05)

            f, mid_v = self.toxicity_model.get_prediction(desired_percentile=0.5, random=False)
            _, upper_v = self.toxicity_model.get_prediction(desired_percentile=0.975, random=False)
            _, lower_v = self.toxicity_model.get_prediction(desired_percentile=0.025, random=False)
            mid_v, upper_v, lower_v = mid_v[:, 1], upper_v[:, 1], lower_v[:, 1]
            ax2.plot(f, mid_v, c='#1f77b4')
            ax2.plot(f, upper_v, c='#1f77b4')
            ax2.plot(f, lower_v, c='#1f77b4')
            ax2.scatter(self.toxicity_model.data_features, self.toxicity_model.data_values,
                        color='black', marker='x', s=50)
            ax2.plot(f, self.environment.toxicity_probabilities, color='black', ls='--')
            ax2.set_ylim(-0.05, 1.05)

            actual_efficacy = self.environment.efficacy_probabilities
            actual_toxicity = self.environment.toxicity_probabilities
            actual_utilities = self.utility_function.get_dose_utility(actual_efficacy, actual_toxicity)

            if exact:
                store_u = np.zeros((np.size(f), 400))
                for k in range(400):
                    _, mid_e = self.efficacy_model.get_prediction(random=True)
                    _, mid_t = self.toxicity_model.get_prediction(random=True)
                    store_u[:, k] = self.utility_function.get_dose_utility(mid_e, mid_t)
                upper_u = np.quantile(store_u, 0.975, 1)
                lower_u = np.quantile(store_u, 0.025, 1)
                mid_pred_u = np.quantile(store_u, 0.5, 1)

            else:
                f, mid_e = self.efficacy_model.get_prediction(desired_percentile=0.5, random=False)
                f, mid_t = self.toxicity_model.get_prediction(desired_percentile=0.5, random=False)
                mid_pred_u = self.utility_function.get_dose_utility(mid_e, mid_t)
                f, high_e = self.efficacy_model.get_prediction(desired_percentile=0.85, random=False)
                f, low_t = self.toxicity_model.get_prediction(desired_percentile=0.15, random=False)
                upper_u = self.utility_function.get_dose_utility(high_e, low_t)
                f, low_e = self.efficacy_model.get_prediction(desired_percentile=0.15, random=False)
                f, high_t = self.toxicity_model.get_prediction(desired_percentile=0.85, random=False)
                lower_u = self.utility_function.get_dose_utility(low_e, high_t)

            ax3.plot(f, actual_utilities, color='black', ls='--')
            ax3.plot(f, mid_pred_u, c='#1f77b4')
            ax3.plot(f, lower_u, c='#1f77b4')
            ax3.plot(f, upper_u, c='#1f77b4')
            ax3.set_ylim(-2, 1.05)

            fig.suptitle(f'1D Utility Maximisation, \n'
                         f' {i} cohorts and {self.block_size * (i)} individuals so far')
            ax1.set_xlabel('Dose')
            ax1.set_ylabel('Efficacy')
            ax2.set_xlabel('Dose')
            ax2.set_ylabel('Toxicity')
            ax3.set_xlabel('Dose')
            ax3.set_ylabel('Utility')
            plt.show()

            self.choose_next_doses()
            self.get_efficacy_data()
            self.get_toxicity_data()
            self.update_efficacy_model()
            self.update_toxicity_model()
            self.update_utility_and_regret()
            self.update_and_clear_data()
            self.suggest_optimal_doses()

    def full_loop_visualise_utility_thompson(self, iterations, exact=False, thompson_iterations=1000):

        actual_efficacy = self.environment.efficacy_probabilities
        actual_toxicity = self.environment.toxicity_probabilities
        actual_utilities = self.utility_function.get_dose_utility(actual_efficacy, actual_toxicity)
        for i in range(iterations):

            plt.figure(figsize=(15, 15))
            ax1 = plt.subplot(2, 3, 1)
            ax2 = plt.subplot(2, 3, 2)
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 1, 2)
            f, mid_v = self.efficacy_model.get_prediction(desired_percentile=0.5, random=False)
            _, upper_v = self.efficacy_model.get_prediction(desired_percentile=0.975, random=False)
            _, lower_v = self.efficacy_model.get_prediction(desired_percentile=0.025, random=False)
            mid_v, upper_v, lower_v = mid_v[:, 1], upper_v[:, 1], lower_v[:, 1]
            ax1.plot(f, mid_v, c='#1f77b4')
            ax1.plot(f, upper_v, c='#1f77b4')
            ax1.plot(f, lower_v, c='#1f77b4')
            ax1.scatter(self.efficacy_model.data_features, self.efficacy_model.data_values,
                        color='black', marker='x', s=50)
            ax1.plot(f, self.environment.efficacy_probabilities, color='black', ls='--')
            ax1.set_ylim(-0.05, 1.05)

            f, mid_v = self.toxicity_model.get_prediction(desired_percentile=0.5, random=False)
            _, upper_v = self.toxicity_model.get_prediction(desired_percentile=0.975, random=False)
            _, lower_v = self.toxicity_model.get_prediction(desired_percentile=0.025, random=False)
            mid_v, upper_v, lower_v = mid_v[:, 1], upper_v[:, 1], lower_v[:, 1]
            ax2.plot(f, mid_v, c='#1f77b4')
            ax2.plot(f, upper_v, c='#1f77b4')
            ax2.plot(f, lower_v, c='#1f77b4')
            ax2.scatter(self.toxicity_model.data_features, self.toxicity_model.data_values,
                        color='black', marker='x', s=50)
            ax2.plot(f, self.environment.toxicity_probabilities, color='black', ls='--')
            ax2.set_ylim(-0.05, 1.05)

            f, mid_e = self.efficacy_model.get_prediction(desired_percentile=0.5, random=False)
            f, mid_t = self.toxicity_model.get_prediction(desired_percentile=0.5, random=False)
            mid_pred_u = self.utility_function.get_dose_utility(mid_e, mid_t)
            f, high_e = self.efficacy_model.get_prediction(desired_percentile=0.85, random=False)
            f, low_t = self.toxicity_model.get_prediction(desired_percentile=0.15, random=False)
            upper_u = self.utility_function.get_dose_utility(high_e, low_t)
            f, low_e = self.efficacy_model.get_prediction(desired_percentile=0.15, random=False)
            f, high_t = self.toxicity_model.get_prediction(desired_percentile=0.85, random=False)
            lower_u = self.utility_function.get_dose_utility(low_e, high_t)

            u_argmax = np.argmax(mid_pred_u)
            ax3.scatter(f[u_argmax], mid_pred_u[u_argmax])

            ax3.plot(f, actual_utilities, color='black', ls='--')
            ax3.plot(f, mid_pred_u, c='#1f77b4')
            ax3.plot(f, lower_u, c='#1f77b4')
            ax3.plot(f, upper_u, c='#1f77b4')
            ax3.set_ylim(-2, 1.05)

            plt.suptitle(f'1D Utility Maximisation, \n'
                         f' {i} cohorts and {self.block_size * (i)} individuals so far')
            ax1.set_xlabel('Dose')
            ax1.set_ylabel('Efficacy')
            ax2.set_xlabel('Dose')
            ax2.set_ylabel('Toxicity')
            ax3.set_xlabel('Dose')
            ax3.set_ylabel('Utility')

            store = np.zeros_like(mid_v)
            for i in range(thompson_iterations):
                _, mid_e = self.efficacy_model.get_prediction(random=True)
                _, mid_t = self.toxicity_model.get_prediction(random=True)
                mid_u = self.utility_function.get_dose_utility(mid_e, mid_t)
                store[uf.randargmax(mid_u)] += 1
            box = np.ones(3) / 3
            y_smooth = np.convolve(store / thompson_iterations, box, mode='same')
            ax4.scatter(f, y_smooth)
            ax4.set_ylim(0)

            ax4.hist(np.asarray(self.dose_data).reshape(-1),
                     bins=np.linspace(0, 1, 21),
                     weights=0.2 * np.ones(len(self.dose_data)) / len(self.dose_data),
                     alpha=0.2)

            plt.show()

            self.choose_next_doses()
            self.get_efficacy_data()
            self.get_toxicity_data()
            self.update_efficacy_model()
            self.update_toxicity_model()
            self.update_utility_and_regret()
            self.update_and_clear_data()
            self.suggest_optimal_doses()

