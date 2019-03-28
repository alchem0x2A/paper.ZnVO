import ase
from ase.neb import NEBTools
from ase.io.trajectory import Trajectory
import os, os.path
from os.path import join, exists, dirname
import numpy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

traj_name = lambda name: join(dirname(__file__), "../../results/NEB/{}V2O5/neb_fin.traj".format(name))


def get_data(name):
    assert name in ("Co", "Zn")
    traj = Trajectory(traj_name(name))
    nb = NEBTools(traj)
    s, E, *_ = nb.get_fit()
    E[-1] = E[0]                # force energy to be same
    ss = numpy.linspace(min(s), max(s))
    ee = interp1d(s, E, kind="cubic")(ss)
    print(max(ee))
    return s, E, ss, ee

plt.style.use("science")

fig = plt.figure(figsize=(3.0, 2.0))
ax = fig.add_subplot(111)
s, E, ss, ee = get_data("Zn")
l, = ax.plot(ss, ee, "-", label="ZVO")
ax.plot(s, E, "o", color=l.get_c())
s, E, ss, ee = get_data("Co")
l,  = ax.plot(ss, ee, "-", label="CVO")
ax.plot(s, E, "o", color=l.get_c())
ax.set_ylim(0, 0.8)
ax.set_xlabel("Diffusion Length ($\\rm{\\AA{}}$)")
ax.set_ylabel("Energy (eV)")
fig.tight_layout()

fig.savefig(join(dirname(__file__), "../../plot/neb/barrier.svg"))
