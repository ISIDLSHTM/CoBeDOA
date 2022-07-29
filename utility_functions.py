class maximise_efficacy():
    def __init__(self):
        pass

    def get_dose_utility(self, efficacy_probabilities, toxicity_probabilities):
        utility = efficacy_probabilities[:, -1].reshape(-1)
        return utility

    def get_individual_utility(self, efficacy_outcome, toxicity_outcome):
        utility = efficacy_outcome
        return utility


class utility_contour():
    def __init__(self, pi1=0.4, pi2=0.7, p=2.07):
        # Default parameters from https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-017-0381-x
        self.pi1 = pi1
        self.pi2 = pi2
        self.p = p

    def get_dose_utility(self, efficacy_probabilities, toxicity_probabilities):
        efficacy_probabilities = efficacy_probabilities[:, -1].reshape(-1)
        toxicity_probabilities = toxicity_probabilities[:, -1].reshape(-1)

        first_part = (1 - efficacy_probabilities) / (1 - self.pi1)
        second_part = toxicity_probabilities / self.pi2
        addition = (first_part ** self.p) + (second_part ** self.p)
        utility = 1 - (addition ** (1 / self.p))
        return utility

    def get_individual_utility(self, efficacy_outcome, toxicity_outcome):
        first_part = (1 - efficacy_outcome) / (1 - self.pi1)
        second_part = toxicity_outcome / self.pi2
        addition = (first_part ** self.p) + (second_part ** self.p)
        utility = 1 - (addition ** (1 / self.p))
        return utility
