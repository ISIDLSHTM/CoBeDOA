import numpy as np
from useful_functions import smooth_1D, smooth_2D, smooth_3D

print('Objective 1 Efficacy')

doses = np.array([-0.5, 1.5]).reshape(-1, 1)
value = np.array([0.05, 0.95]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 11, 21, xlim=(-1, 2), xres=301)
doses = doses[100:201]
values = values[100:201]
np.save('Environment_1_101_1_d.npy', doses)
np.save('Environment_1_101_1_v.npy', values)

doses = np.array([0, 1]).reshape(-1, 1)
value = np.array([0.05, 0.9]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 21, 9, xres=101)
np.save('Environment_1_101_2_d.npy', doses)
np.save('Environment_1_101_2_v.npy', values)

doses = np.array([-0.2, .1, .5, .9, 1.2]).reshape(-1, 1)
value = np.array([0.1, 0.5, 0.8, 0.55, 0.2]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 21, 15, xlim=(-0.5, 1.5), xres=201)
doses = doses[50:151]
values = values[50:151]
np.save('Environment_1_101_3_d.npy', doses)
np.save('Environment_1_101_3_v.npy', values)

doses = np.array([0, .2, .4, .6, .7, 1]).reshape(-1, 1)
value = np.array([0.05, 0.1, 0.15, 0.3, 0.7, 0.5]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 21, 5, xres=101)
np.save('Environment_1_101_4_d.npy', doses)
np.save('Environment_1_101_4_v.npy', values)

doses = np.array([0, .2, .4, .6, .8, 1]).reshape(-1, 1)
value = np.array([.8, .7, .3, .15, .1, 0.05]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 21, 5, xres=101)
np.save('Environment_1_101_5_d.npy', doses)
np.save('Environment_1_101_5_v.npy', values)

doses = np.array([0, .2, .55, .65, 1]).reshape(-1, 1)
value = np.array([.3, .5, .7, .50, .7]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 21, 5, xres=101)
np.save('Environment_1_101_6_d.npy', doses)
np.save('Environment_1_101_6_v.npy', values)

doses = np.array([0, .5, 1]).reshape(-1, 1)
value = np.array([0.75, 0.85, .75]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 11, 11, xlim=(0, 1), xres=101)
np.save('Environment_1_101_7_d.npy', doses)
np.save('Environment_1_101_7_v.npy', values)

print('Objective 2 Efficacy')

data = np.asarray([[1, 1], [0, 0], [0, 1], [1, 0], [.6, .6]])
value = np.asarray([[.5], [.1], [.4], [.4], [.9]])
p, v = smooth_2D(data, value, 9, 15, xres=21, yres=21)
np.save('Environment_2_441_1_d.npy', p)
np.save('Environment_2_441_1_v.npy', v)

data = np.asarray([[1, 1], [0, 0], [0, 1], [1, 0]])
value = np.asarray([[.5], [.1], [.8], [.9]])
p, v = smooth_2D(data, value, 11, 9, xres=21, yres=21)
np.save('Environment_2_441_2_d.npy', p)
np.save('Environment_2_441_2_v.npy', v)

data = np.asarray([[1, 1], [0, 0], [0, 1], [1, 0]])
value = np.asarray([[.95], [.05], [.6], [.6]])
p, v = smooth_2D(data, value, 11, 9, xres=21, yres=21)
np.save('Environment_2_441_3_d.npy', p)
np.save('Environment_2_441_3_v.npy', v)

data = np.asarray([[1, 1], [.9, .9], [0, 0], [0, 1], [1, 0]])
value = np.asarray([[.6], [.95], [.05], [.4], [.6]])
p, v = smooth_2D(data, value, 11, 9, xres=21, yres=21)
np.save('Environment_2_441_4_d.npy', p)
np.save('Environment_2_441_4_v.npy', v)

data = np.asarray([[1, 1], [1, 0.5], [1, 0],
                   [.9, 1], [.9, 0.5], [.9, 0],
                   [0, 1], [0, 0.5], [0, 0]])
value = np.asarray([[.8], [.7], [.7],
                    [.9], [.8], [.7],
                    [.1], [.05], [.05]])
p, v = smooth_2D(data, value, 10, 9, xres=21, yres=21)
np.save('Environment_2_441_5_d.npy', p)
np.save('Environment_2_441_5_v.npy', v)

data = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                   [.7, .1, .3], [.7, .1, .3], [.7, .1, .3]])
value = np.asarray([[0], [0], [0], [0],
                    [0], [0], [0], [0],
                    [.9], [.9], [.9]])
p, v = smooth_3D(data, value, 5, 6, xres=11, yres=11, zres=11)
np.save('Environment_2_1331_6_d.npy', p)
np.save('Environment_2_1331_6_v.npy', v)

data = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                   [.5, 0, 0],
                   [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                   [.5, 0, 0]
                   ])
value = np.asarray([[0.1], [0.5], [0.5], [0.8],
                    [0.1], [0.9], [0.9], [0.4],
                    [0.1],
                    [0.1], [0.5], [0.5], [0.8],
                    [0.1], [0.9], [0.9], [0.4],
                    [0.1]
                    ])
p, v = smooth_3D(data, value, iterations=3, m=27, xres=11, yres=11, zres=11)
np.save('Environment_l_1331_7_d.npy', p)
np.save('Environment_l_1331_7_v.npy', v)

print('Toxicity')

doses = np.array([0.2, 1.5]).reshape(-1, 1)
value = np.array([0.05, 0.65]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 11, 21, xlim=(-1, 2), xres=301)
doses = doses[100:201]
values = values[100:201]
np.save('Environment_4_101_1_d.npy', doses)
np.save('Environment_4_101_1_v.npy', values)

doses = np.array([0.7, .8]).reshape(-1, 1)
value = np.array([0.05, 0.95]).reshape(-1, 1)
doses, values = smooth_1D(doses, value, 11, 3, xlim=(-1, 2), xres=301)
doses = doses[100:201]
values = values[100:201]
np.save('Environment_4_101_2_d.npy', doses)
np.save('Environment_4_101_2_v.npy', values)

data = np.asarray([[1, 1], [0, 0], [0, 1], [1, 0],
                   [.5, .5], [0, .5], [.5, 0],
                   [1, .5], [.5, 1]])
value = np.asarray([[.9], [.1], [.7], [.8],
                    [.2], [.23], [.2],
                    [.85], [.75]])
p, v = smooth_2D(data, value, 9, 15, xres=21, yres=21)
np.save('Environment_4_441_3_d.npy', p)
np.save('Environment_4_441_3_v.npy', v)
