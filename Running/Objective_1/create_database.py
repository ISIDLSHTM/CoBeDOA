import sqlite3

conn = sqlite3.connect('__Objective_1_Database.db')
c = conn.cursor()

c.execute("""CREATE TABLE experiments (
            experiment_id int32,
            method_name str,
            environment_name str
            )""")

c.execute("""CREATE TABLE individuals (
            experiment_id int32,
            individual_number int16,
            dose str,
            efficacy str,
            toxicity str,
            predicted_optimal_dose str,
            actual_utility_at_predicted float32,
            inaccuracy float32,
            individual_regret float32,
            cumulative_regret float32,
            average_regret float32
            )""")

conn.close()
