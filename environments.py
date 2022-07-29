from environment_class import Environment
import numpy as np
import pathlib

data_path = str(pathlib.Path(__file__).parent.resolve()) + '\pickling_for_environments'

d = np.load(data_path + '\Environment_1_101_1_d.npy')
e = np.load(data_path + '\Environment_1_101_1_v.npy')
t = e * 0
Environment_1_101_1 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_2_d.npy')
e = np.load(data_path + '\Environment_1_101_2_v.npy')
t = e * 0
Environment_1_101_2 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_3_d.npy')
e = np.load(data_path + '\Environment_1_101_3_v.npy')
t = e * 0
Environment_1_101_3 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_4_d.npy')
e = np.load(data_path + '\Environment_1_101_4_v.npy')
t = e * 0
Environment_1_101_4 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_5_d.npy')
e = np.load(data_path + '\Environment_1_101_5_v.npy')
t = e * 0
Environment_1_101_5 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_6_d.npy')
e = np.load(data_path + '\Environment_1_101_6_v.npy')
t = e * 0
Environment_1_101_6 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_7_d.npy')
e = np.load(data_path + '\Environment_1_101_7_v.npy')
t = e * 0
Environment_1_101_7 = Environment(d, e, t)

"""
Objective 2
"""

d = np.load(data_path + '\Environment_2_441_1_d.npy')
e = np.load(data_path + '\Environment_2_441_1_v.npy')
t = e * 0
Environment_2_441_1 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_441_2_d.npy')
e = np.load(data_path + '\Environment_2_441_2_v.npy')
t = e * 0
Environment_2_441_2 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_441_3_d.npy')
e = np.load(data_path + '\Environment_2_441_3_v.npy')
t = e * 0
Environment_2_441_3 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_441_4_d.npy')
e = np.load(data_path + '\Environment_2_441_4_v.npy')
t = e * 0
Environment_2_441_4 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_441_5_d.npy')
e = np.load(data_path + '\Environment_2_441_5_v.npy')
t = e * 0
Environment_2_441_5 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_1331_6_d.npy')
e = np.load(data_path + '\Environment_2_1331_6_v.npy')
t = e * 0
Environment_2_1331_6 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_1331_7_d.npy')
e = np.load(data_path + '\Environment_2_1331_7_v.npy')
t = e * 0
Environment_2_1331_7 = Environment(d, e, t)

"""
Objective 3
"""

d = np.load(data_path + '\Environment_1_101_1_d.npy')
e = np.load(data_path + '\Environment_1_101_1_v.npy')
t = np.load(data_path + '\Environment_4_101_1_v.npy')
Environment_3_101_1 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_4_d.npy')
e = np.load(data_path + '\Environment_1_101_4_v.npy')
t = np.load(data_path + '\Environment_4_101_1_v.npy')
Environment_3_101_2 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_1_d.npy')
e = np.load(data_path + '\Environment_1_101_1_v.npy')
t = np.load(data_path + '\Environment_4_101_2_v.npy')
Environment_3_101_3 = Environment(d, e, t)

d = np.load(data_path + '\Environment_1_101_4_d.npy')
e = np.load(data_path + '\Environment_1_101_4_v.npy')
t = np.load(data_path + '\Environment_4_101_2_v.npy')
Environment_3_101_4 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_441_1_d.npy')
e = np.load(data_path + '\Environment_2_441_3_v.npy')
t = np.load(data_path + '\Environment_4_441_3_v.npy')
Environment_3_441_5 = Environment(d, e, t)

d = np.load(data_path + '\Environment_2_441_2_d.npy')
e = np.load(data_path + '\Environment_2_441_2_v.npy')
t = np.load(data_path + '\Environment_4_441_3_v.npy')
Environment_3_441_6 = Environment(d, e, t)
