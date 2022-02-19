import argparse
##from matplotlib import pyplot as plt

import numpy as np
import pyscf
import pyscf.pbc

import vayesta
import vayesta.misc
#import vayesta.misc.plot_mol
from vayesta.misc.plotting.plotly_mol import plot_mol

from structures import NO2_Graphene


parser = argparse.ArgumentParser()
parser.add_argument('--structure', type=int, default=-1)
parser.add_argument('--supercell', type=int, nargs=2, default=[5, 5])
parser.add_argument('--popfile')
parser.add_argument('--popfile-columns', type=int, nargs='*', default=[3, 5])
parser.add_argument('-o')
args = parser.parse_args()

if args.o is None:
    args.o = '%s.html' % args.popfile

cell = pyscf.pbc.gto.Cell()
no2_graphene = NO2_Graphene(args.supercell, structure=args.structure)
cell.a, cell.atom = no2_graphene.amat, no2_graphene.atom
cell.dimension = 2
cell.spin = 1
cell.build()


data = np.loadtxt(args.popfile, usecols=(0, *args.popfile_columns))
charges = data[:,1]
spins = data[:,2]
charges[:3] = 0
spins[:3] = 0

fig = plot_mol(cell, charges=charges, spins=spins, add_images=True)
fig.write_html(args.o)

#plt.show()
