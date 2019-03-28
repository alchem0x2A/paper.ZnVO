import numpy
from ase.io.cube import read_cube_data
from ase.build import make_supercell
from ase.visualize.plot import plot_atoms
import os, os.path
from os.path import exists, join, dirname


import matplotlib.pyplot as plt
plt.style.use("science")

from scipy.interpolate import griddata


img_path = join(dirname(__file__), "../../plot/potential")
data_path = join(dirname(__file__), "../../results/potential_energy")

if not exists(img_path):
    os.makedirs(img_path)

cube_file = lambda name: join(data_path, "{}.cube".format(name))

def plot(name):
    assert name in ("ZnVO", "CoVO")
    data, atoms = read_cube_data(cube_file(name))

    lattice = atoms.cell
    na, nb, nc = data.shape
    layer = data[:, :, nc // 2].T
    print(layer.max(), layer.min())
    x_ = numpy.linspace(0, 1, na)
    y_ = numpy.linspace(0, 1, nb)
    xx_s, yy_s = numpy.meshgrid(x_, y_)
    xx_fine, yy_fine = numpy.meshgrid(numpy.linspace(0, 1, 256), numpy.linspace(0, 1, 256))
    
    xy_flat = numpy.vstack([xx_s.flat, yy_s.flat]).T
    print(xy_flat.shape)
    c_fine = griddata(xy_flat, layer.flat, (xx_fine, yy_fine), method="cubic")
    zz_fine = numpy.ones_like(yy_fine) * 0.5
    xx, yy, zz = numpy.tensordot(lattice.T,
                                 numpy.array([xx_fine, yy_fine, zz_fine]),
                                 axes=1)
    # print(xx, yy)
    fig = plt.figure(figsize=(3.5, 3.0))
    ax = fig.add_subplot(111)
    # layer = layer.T
    ax.pcolor(xx, yy, c_fine,
              cmap="rainbow_r",
              vmin=-10,
              # antialiased=True,
              # interpolate="bicubic",
              rasterized=True)
    sc = make_supercell(atoms, numpy.diag([2, 2, 1]))
    idx = [p.index for p in sc if (p.z > atoms.cell[-1, -1]*0.43) \
                                   and (p.z < atoms.cell[-1, -1] * 0.58)]
                                   # and (p.x < atoms.cell[0, 0] * 1.1) \
                                   # and (p.y < atoms.cell[1, 1] * 1.2)]
    new_atoms = sc[idx]
    rot = {"ZnVO": (-0.4, 1.30, 0.02),
           "CoVO": (0.00, 1.89, 0.00)}
    off = {"ZnVO": (-1.40, -1.15),
           "CoVO": (-0.9, -1.2)}
    plot_atoms(new_atoms, ax=ax, rotation="{0}x,{1}y,{2}z".format(*rot[name]),
               radii=0.6,
               offset=off[name])
               # show_unit_cell=True)
               # offset=(-1.1, -0.8),
               # show_unit_cell=True)
    # plt.show()
    ax.set_aspect("equal")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_axis_off()
    # ax.set_axis_off()
    fig.savefig(join(img_path, "{}_half.svg".format(name)))
    
    



    
if __name__ == "__main__":
    plot("ZnVO")
    plot("CoVO")
