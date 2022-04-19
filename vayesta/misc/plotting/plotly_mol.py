import string
import itertools
import numbers

import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import pyscf
import pyscf.pbc

from .colors import get_atom_color


def mol_supercell(mol, charges, spins):
    if not hasattr(mol, 'lattice_vectors'):
        return mol
    if mol.dimension == 1:
        images = [3, 1, 1]
        #images = [1, 0, 0]
    elif mol.dimension == 2:
        images = [3, 3, 1]
        #images = [1, 1, 0]
    else:
        images = [3, 3, 3]
        #images = [1, 1, 1]
    mol = pyscf.pbc.tools.super_cell(mol, images)
    #mol = pyscf.pbc.tools.cell_plus_imgs(mol, images)
    #nimages = images[0]*images[1]*images[2]
    #ncells = np.product(2*np.asarray(images)+1)
    ncells = np.product(images)
    if charges is not None:
        charges = ncells * list(charges)
    if spins is not None:
        spins = ncells * list(spins)
    return mol, charges, spins

def get_bonds(mol, threshold=3.0):
    coords = mol.atom_coords()
    bonds_x = []
    bonds_y = []
    bonds_z = []
    for atom, coord in enumerate(coords):
        for atom2, coord2 in enumerate(coords[:atom]):
            distance = np.linalg.norm(coord - coord2)
            if distance > threshold:
                continue
            bonds_x += [coord[0], coord2[0], None]
            bonds_y += [coord[1], coord2[1], None]
            bonds_z += [coord[2], coord2[2], None]
    return bonds_x, bonds_y, bonds_z

def get_cell_bounds(mol):
    amat = mol.lattice_vectors()
    points_x = []
    points_y = []
    points_z = []

    def add_line(point1, point2):
        nonlocal points_x, points_y, points_z
        points_x += [point1[0], point2[0], None]
        points_y += [point1[1], point2[1], None]
        points_z += [point1[2], point2[2], None]

    corner = [np.asarray((0, 0, 0)), amat[0], amat[1], amat[2],
            amat[0]+amat[1], amat[0]+amat[2], amat[1]+amat[2], amat[0]+amat[1]+amat[2]]

    add_line(corner[0], corner[1])
    add_line(corner[0], corner[2])
    add_line(corner[0], corner[3])

    add_line(corner[1], corner[4])
    add_line(corner[1], corner[5])

    add_line(corner[2], corner[4])
    add_line(corner[2], corner[6])

    add_line(corner[3], corner[5])
    add_line(corner[3], corner[6])

    add_line(corner[4], corner[7])
    add_line(corner[5], corner[7])
    add_line(corner[6], corner[7])
    return points_x, points_y, points_z

def get_ranges(mol, margin=1.0):

    #xmin, xmax = coords[:,0].min(), coords[:,0].max()
    #ymin, ymax = coords[:,1].min(), coords[:,1].max()
    #zmin, zmax = coords[:,2].min(), coords[:,2].max()

    amat = mol.lattice_vectors()
    m = amat[0]+amat[1]+amat[2]

    xmin, xmax = -margin, m[0]+margin
    ymin, ymax = -margin, m[1]+margin
    zmin, zmax = -margin, m[2]+margin
    return [xmin, xmax], [ymin, ymax], [zmin, zmax]


def plot_mol(mol, charges=None, spins=None, add_images=False, **kwargs):

    mol0 = mol
    if add_images:
        mol, charges, spins = mol_supercell(mol, charges, spins)

    atoms = mol._atom
    #a_matrix = mol.lattice_vectors() if hasattr(mol, 'lattice_vectors') else None
    coords = mol.atom_coords()
    symbols = [mol.atom_symbol(a) for a in range(len(atoms))]
    atom_colors = [get_atom_color(s) for s in symbols]

    # Bonds
    bonds_x, bonds_y, bonds_z = get_bonds(mol)
    bonds = go.Scatter3d(x=bonds_x, y=bonds_y, z=bonds_z, mode='lines', line=dict(width=5, color='grey'))
    data = [bonds]

    # Bounds
    bounds_x, bounds_y, bounds_z = get_cell_bounds(mol0)
    bounds = go.Scatter3d(x=bounds_x, y=bounds_y, z=bounds_z, mode='lines', line=dict(width=5, color='rgb(0, 200, 0)'))
    data.append(bounds)
    #data.append(bonds)

    def make_scatter(colors, colorscale, **kwargs):
        colorbar = dict(thickness=20) if colorscale else None
        marker = go.scatter3d.Marker(size=8, color=colors, colorscale=colorscale, cmid=0.0, colorbar=colorbar, line=dict(width=4, color='black'))
        hovertext = ['Atom %d: %s' % (i, s) for (i, s) in enumerate(symbols)]
        scatter = go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], marker=marker, mode='markers', text=hovertext, **kwargs)
        return scatter

    scatter = make_scatter(atom_colors, None)
    data.append(scatter)
    if charges is not None:
        scatter = make_scatter(charges, ['Blue', 'White', 'Red'], visible=False)
        data.append(scatter)
    if spins is not None:
        scatter = make_scatter(spins, ['Blue', 'White', 'Red'], visible=False)
        data.append(scatter)

    range_x, range_y, range_z = get_ranges(mol)
    scene = dict(aspectmode='data',
            xaxis = dict(backgroundcolor="rgb(220, 220, 220)", gridcolor='black', zerolinecolor='black', range=range_x),
            yaxis = dict(backgroundcolor="rgb(200, 200, 200)", gridcolor='black', zerolinecolor='black', range=range_y),
            zaxis = dict(backgroundcolor="rgb(180, 180, 180)", gridcolor='black', zerolinecolor='black', range=range_z),
            )


    bonds_bounds = [True, True]
    buttons = [
        dict(label='Atoms', method='update', args=[{'visible':   bonds_bounds+[True, False, False]}]),
	dict(label='Charges', method='update', args=[{'visible': bonds_bounds+[False, True, False]}]),
	dict(label='Spins', method='update', args=[{'visible':   bonds_bounds+[False, False, True]}])
	]
    updatemenus = list([dict(active=0, buttons=buttons)])

    layout = go.Layout(scene=scene, showlegend=False, updatemenus=updatemenus)
    fig=go.Figure(data=data, layout=layout)

    return fig


if __name__ == '__main__':
    import pyscf
    import pyscf.gto

    mol = pyscf.gto.Mole()
    mol.atom = 'H 0 0 0 ; F 0 0 2'
    mol.build()

    charges = [-1.0, 0.5]
    spins = [-1, 1]
    #colors = None
    fig = plot_mol(mol, charges=charges, spins=spins)
    fig.write_html('test.html')
