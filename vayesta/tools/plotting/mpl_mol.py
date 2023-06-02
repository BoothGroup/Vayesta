import string
import itertools
import numbers

import numpy as np
from matplotlib import pyplot as plt
import pyscf.pbc

from vayesta.tools.plotting.colors import atom_colors


def plot_mol(mol, size=30, indices=False, colors=None, colormap='bwr', add_images=False, **kwargs):

    if add_images and hasattr(mol, 'lattice_vectors'):
        if mol.dimension == 1:
            images = [3, 1, 1]
        elif mol.dimension == 2:
            images = [3, 3, 1]
        else:
            images = [3, 3, 3]
        mol = pyscf.pbc.tools.super_cell(mol, images)
        if colors is not None:
            nimages = images[0]*images[1]*images[2]
            colors = nimages*list(colors)

    atoms = mol._atom
    a_matrix = mol.lattice_vectors() if hasattr(mol, 'lattice_vectors') else None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if a_matrix is not None:
        x = np.zeros((1,))
        y = np.zeros((1,))
        z = np.zeros((1,))
        axcolors = ["red", "blue", "green"]
        for d in range(3):
            #dx=a_matrix[d,0]
            #dy=a_matrix[d,1]
            #dz=a_matrix[d,2]
            dx, dy, dz = a_matrix[d]
            ax.quiver(x, y, z, dx, dy, dz, color=axcolors[d])
        ax.set_xlim(1.1*a_matrix[0,0])
        ax.set_ylim(1.1*a_matrix[1,1])
        ax.set_zlim(1.1*a_matrix[2,2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    x = [a[1][0] for a in atoms]
    y = [a[1][1] for a in atoms]
    z = [a[1][2] for a in atoms]

    sym = []
    for a in atoms:
        s = ""
        for l in a[0]:
            if l in string.ascii_letters:
                s += l
        sym.append(s)
    #sym = [re.sub("\d", "", a[0])  for a in atoms]
    if colors is None:
        colors = [atom_colors.get(s, "black") for s in sym]

    if isinstance(colors[0], numbers.Number):
        vmax = np.max(np.abs(colors))
        kwargs['vmax'] = kwargs.get('vmax', vmax)
        kwargs['vmin'] = kwargs.get('vmin', -vmax)

    points = ax.scatter(x, y, z, s=size, c=colors, depthshade=False, edgecolor="black", cmap=colormap, **kwargs)
    #ax.set_box_aspect((1,1,1))
    # NotImplemented...
    #ax.set_aspect('equal')
    if isinstance(colors[0], numbers.Number):
        fig.colorbar(points)



    maxdist = np.amax((x, y, z))
    for i, j, k in itertools.product([-1, 1], repeat=3):
        ax.scatter(i*maxdist, j*maxdist, k*maxdist, color="w", alpha=0.0)

    if indices:
        offset = [0, 0, 1]
        for i, atom in enumerate(atoms):
            idx = atom[0][-1]
            idx = idx if idx.isdigit() else None
            if idx is not None:
                ax.text(x[i]+offset[0], y[i]+offset[1], z[i]+offset[2], idx)

    return fig


if __name__ == '__main__':
    import pyscf.gto

    mol = pyscf.gto.Mole()
    mol.atom = 'H 0 0 0 ; F 0 0 2'
    mol.build()

    #fig = plot_mol(mol, colors=['red', 'green'])
    fig = plot_mol(mol, colors=[0, 3], colormap='bwr')
    plt.show()
