import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

import pathlib

data_path = str(pathlib.Path(__file__).parent.resolve()) + '\pickling_for_environments'

print(data_path)

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v']
fignum, obj, idx = 6, 1, 1

d = np.load(data_path + '\Environment_1_101_1_d.npy')
e = np.load(data_path + '\Environment_1_101_1_v.npy')
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
idx += 1
plt.close()

d = np.load(data_path + '\Environment_1_101_2_d.npy')
e = np.load(data_path + '\Environment_1_101_2_v.npy')
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
idx += 1
plt.close()

d = np.load(data_path + '\Environment_1_101_3_d.npy')
e = np.load(data_path + '\Environment_1_101_3_v.npy')
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
idx += 1
plt.close()

d = np.load(data_path + '\Environment_1_101_4_d.npy')
e = np.load(data_path + '\Environment_1_101_4_v.npy')
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
idx += 1
plt.close()

d = np.load(data_path + '\Environment_1_101_5_d.npy')
e = np.load(data_path + '\Environment_1_101_5_v.npy')
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
idx += 1
plt.close()

d = np.load(data_path + '\Environment_1_101_6_d.npy')
e = np.load(data_path + '\Environment_1_101_6_v.npy')
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
idx += 1
plt.close()

d = np.load(data_path + '\Environment_1_101_7_d.npy')
e = np.load(data_path + '\Environment_1_101_7_v.npy')
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
idx += 1
plt.close()

#
# --------------------------------------------------------------------#
# Objective 2
# --------------------------------------------------------------------#

fignum, obj, idx = 7, 2, 1

d = np.load(data_path + '\Environment_2_441_1_d.npy')
e = np.load(data_path + '\Environment_2_441_1_v.npy')
x = d[:, 0].reshape(-1)
y = d[:, 1].reshape(-1)
z = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Efficacy')
fig.colorbar(p)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
plt.close()
idx += 1

d = np.load(data_path + '\Environment_2_441_2_d.npy')
e = np.load(data_path + '\Environment_2_441_2_v.npy')
x = d[:, 0].reshape(-1)
y = d[:, 1].reshape(-1)
z = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Efficacy')
fig.colorbar(p)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
plt.close()
idx += 1

d = np.load(data_path + '\Environment_2_441_3_d.npy')
e = np.load(data_path + '\Environment_2_441_3_v.npy')
x = d[:, 0].reshape(-1)
y = d[:, 1].reshape(-1)
z = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Efficacy')
fig.colorbar(p)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
plt.close()
idx += 1

d = np.load(data_path + '\Environment_2_441_4_d.npy')
e = np.load(data_path + '\Environment_2_441_4_v.npy')
x = d[:, 0].reshape(-1)
y = d[:, 1].reshape(-1)
z = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Efficacy')
fig.colorbar(p)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
plt.close()
idx += 1

d = np.load(data_path + '\Environment_2_441_5_d.npy')
e = np.load(data_path + '\Environment_2_441_5_v.npy')
x = d[:, 0].reshape(-1)
y = d[:, 1].reshape(-1)
z = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Efficacy')
fig.colorbar(p)
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
plt.close()
idx += 1

cmap = pl.cm.cool
my_cmap = cmap(np.arange(cmap.N))
alphas = np.linspace(0, 1, cmap.N)
BG = np.asarray([1., 1., 1., ])
for i in range(cmap.N):
    my_cmap[i, :-1] = my_cmap[i, :-1] * alphas[i] + BG * (1. - alphas[i])
my_cmap = ListedColormap(my_cmap)

d = np.load(data_path + '\Environment_2_1331_6_d.npy')
e = np.load(data_path + '\Environment_2_1331_6_v.npy')
x = d[:, 1].reshape(-1)
y = d[:, 2].reshape(-1)
z = d[:, 0].reshape(-1)
c = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=c, cmap=my_cmap, vmin=0, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Dose dimension 3')
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
plt.close()
idx += 1

d = np.load(data_path + '\Environment_2_1331_7_d.npy')
e = np.load(data_path + '\Environment_2_1331_7_v.npy')
x = d[:, 1].reshape(-1)
y = d[:, 2].reshape(-1)
z = d[:, 0].reshape(-1)
c = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=c, cmap=my_cmap, vmin=0, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[idx - 1]}, Objective {obj}, Scenario {idx}')
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
ax.invert_zaxis()
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Dose dimension 3')
plt.savefig(f'Environment_plots/{obj}_{idx}.png')
plt.close()

# # #--------------------------------------------------------------------#
# # Objective 3
# # #--------------------------------------------------------------------#
from utility_functions import utility_contour

uf = utility_contour()

fignum, obj, idx, al = 8, 3, 1, 0

d = np.load(data_path + '\Environment_1_101_1_d.npy')
e = np.load(data_path + '\Environment_1_101_1_v.npy')
t = np.load(data_path + '\Environment_4_101_1_v.npy')
u = uf.get_dose_utility(e, t)
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Efficacy')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_1_eff')
plt.close()
plt.plot(d, t)
plt.xlabel('Dose')
plt.ylabel('Toxicity')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Toxicity')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_1_tox')
plt.close()
plt.plot(d, u)
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Utility')
al += 1
idx += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.savefig('environment_plots/4_1_uti')
plt.close()

d = np.load(data_path + '\Environment_1_101_4_d.npy')
e = np.load(data_path + '\Environment_1_101_4_v.npy')
t = np.load(data_path + '\Environment_4_101_1_v.npy')
u = uf.get_dose_utility(e, t)
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Efficacy')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_2_eff')
plt.close()
plt.plot(d, t)
plt.xlabel('Dose')
plt.ylabel('Toxicity')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Toxicity')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_2_tox')
plt.close()
plt.plot(d, u)
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Utility')
al += 1
idx += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.savefig('environment_plots/4_2_uti')
plt.close()

d = np.load(data_path + '\Environment_1_101_1_d.npy')
e = np.load(data_path + '\Environment_1_101_1_v.npy')
t = np.load(data_path + '\Environment_4_101_2_v.npy')
u = uf.get_dose_utility(e, t)
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Efficacy')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_3_eff')
plt.close()
plt.plot(d, t)
plt.xlabel('Dose')
plt.ylabel('Toxicity')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Toxicity')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_3_tox')
plt.close()
plt.plot(d, u)
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Utility')
al += 1
idx += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.savefig('environment_plots/4_3_uti')
plt.close()

d = np.load(data_path + '\Environment_1_101_4_d.npy')
e = np.load(data_path + '\Environment_1_101_4_v.npy')
t = np.load(data_path + '\Environment_4_101_2_v.npy')
u = uf.get_dose_utility(e, t)
plt.plot(d, e)
plt.xlabel('Dose')
plt.ylabel('Efficacy')
plt.ylim(0, 1)
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Efficacy')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_4_eff')
plt.close()
plt.plot(d, t)
plt.xlabel('Dose')
plt.ylabel('Toxicity')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Toxicity')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('environment_plots/4_4_tox')
plt.close()
plt.plot(d, u)
plt.xlabel('Dose')
plt.ylabel('Utility')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Utility')
al += 1
idx += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.savefig('environment_plots/4_4_uti')
plt.close()
#
#
d = np.load(data_path + '\Environment_2_441_1_d.npy')
e = np.load(data_path + '\Environment_2_441_3_v.npy')
t = np.load(data_path + '\Environment_4_441_3_v.npy')
u = uf.get_dose_utility(e, t)
x = d[:, 0].reshape(-1)
y = d[:, 1].reshape(-1)
z = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Efficacy')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Efficacy')
plt.savefig('environment_plots/4_5_eff')
plt.close()
z = t.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Toxicity')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Toxicity')
plt.savefig('environment_plots/4_5_tox')
plt.close()
z = u.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=-1, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Utility')
al += 1
idx += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(-1, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Utility')
plt.savefig('environment_plots/4_5_uti')
plt.close()

d = np.load(data_path + '\Environment_2_441_2_d.npy')
e = np.load(data_path + '\Environment_2_441_2_v.npy')
t = np.load(data_path + '\Environment_4_441_3_v.npy')
u = uf.get_dose_utility(e, t)
x = d[:, 0].reshape(-1)
y = d[:, 1].reshape(-1)
z = e.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Efficacy')
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Efficacy')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.savefig('environment_plots/4_6_eff')
plt.close()
z = t.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=0, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Toxicity')
al += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(0, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Toxicity')
plt.savefig('environment_plots/4_6_tox')
plt.close()
z = u.reshape(-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("cool")
p = ax.scatter3D(x, y, z, z, c=z, cmap=cmhot, vmin=-1, vmax=1)
fig.colorbar(p)
ax.view_init(elev=45., azim=225)
plt.title(f'Figure {fignum}{alphabet[al]}, Objective {obj}, Scenario {idx} Utility')
al += 1
idx += 1
plt.tight_layout()
plt.xlim(0, 1)
plt.ylim(0, 1)
ax.set_zlim(-1, 1)
plt.xlabel('Dose dimension 1')
plt.ylabel('Dose dimension 2')
ax.set_zlabel('Utility')
plt.savefig('environment_plots/4_6_uti')
plt.close()