class Individual():
    def __init__(self, Experiment_ID, Individual_Number,
                 Dose, Efficacy, Toxicity,
                 Predicted_Optimal_Dose,
                 Predicted_Utility, Actual_Utility_At_Predicted, Inaccuracy,
                 Individual_Regret, Cumulative_Regret, Average_Regret,
                 ):
        self.Average_Regret = Average_Regret
        self.Cumulative_Regret = Cumulative_Regret
        self.Individual_Regret = Individual_Regret
        self.Inaccuracy = Inaccuracy
        self.Actual_Utility_At_Predicted = Actual_Utility_At_Predicted
        self.Predicted_Utility = Predicted_Utility
        self.Predicted_Optimal_Dose = Predicted_Optimal_Dose
        self.Dose = Dose
        self.Toxicity = Toxicity
        self.Efficacy = Efficacy
        self.Individual_Number = Individual_Number
        self.Experiment_ID = Experiment_ID


class Experiment_Record():
    def __init__(self, Experiment_ID, Method_Name, Environment_Name):
        self.Environment_Name = Environment_Name
        self.Method_Name = Method_Name
        self.Experiment_ID = Experiment_ID
