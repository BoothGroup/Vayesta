# Standard
import sys
import argparse
import os.path
from datetime import datetime
# External
import numpy as np
# Internal
import pyscf
import pyscf.lib
import pyscf.lo
import pyscf.scf
import pyscf.pbc
import pyscf.pbc.dft
import pyscf.pbc.tools
import pyscf.pbc.df
import pyscf.pbc.mp
import pyscf.pbc.cc

import vayesta
import vayesta.ewf
from vayesta.misc import counterpoise
from vayesta.misc import molstructs
log = vayesta.log

# Olli's incore GDF
try:
    from pyscf.pbc.df.df_incore import IncoreGDF
except:
    log.error("IncoreGDF not found")
    IncoreGDF = pyscf.pbc.df.GDF

try:
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    timer = MPI.Wtime
except (ImportError, ModuleNotFoundError) as e:
    MPI = False
    MPI_rank = 0
    MPI_size = 1
    from timeit import default_timer as timer

def str_or_none(s):
    if s.lower() in ["none", ""]:
        return None
    return s

default_auxbasis = {
        'def2-svp' : 'def2-svp-ri',
        'def2-tzvp' : 'def2-tzvp-ri',
        'def2-qzvp' : 'def2-qzvp-ri',
        }

def get_arguments():
    """Get arguments from command line."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # System
    parser.add_argument("--system", choices=["diamond", "graphite", "graphene", "hbn", "perovskite", 'SrTiO3', 'SrTiO3-I4'], default="graphene")
    parser.add_argument("--atoms", nargs="*")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--pseudopot", type=str_or_none)
    parser.add_argument("--ecp")
    parser.add_argument("--supercell", type=int, nargs=3)
    parser.add_argument("--k-points", type=int, nargs=3)
    parser.add_argument("--lattice-consts", type=float, nargs="*")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--only", type=int, nargs="*")
    parser.add_argument("--ndim", type=int)
    parser.add_argument("--vacuum-size", type=float)                    # For 2D
    parser.add_argument("--precision", type=float, default=1e-8)
    parser.add_argument("--pyscf-verbose", type=int, default=10)
    parser.add_argument("--exp-to-discard", type=float, help="If set, discard diffuse basis functions.")
    # Counterpoise
    parser.add_argument("--counterpoise")
    parser.add_argument("--counterpoise-nimages", type=int, default=1)
    parser.add_argument("--counterpoise-rmax", type=float, default=np.inf)
    parser.add_argument("--counterpoise-unit", default='a0')
    # Mean-field
    parser.add_argument("--save-scf", help="Save primitive cell SCF.", default="scf-%.2f.chk")           # If containg "%", it will be formatted as args.save_scf % a with a being the lattice constant
    parser.add_argument("--load-scf", help="Load primitive cell SCF.")
    parser.add_argument("--load-scf-init-guess")
    parser.add_argument("--exxdiv-none", action="store_true")
    parser.add_argument("--hf-init-guess-basis")
    parser.add_argument("--scf-conv-tol", type=float)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    #parser.add_argument("--remove-linear-dep", type=float)
    parser.add_argument("--lindep-threshold", type=float)
    # Density-fitting
    parser.add_argument("--df", choices=["FFTDF", "GDF", "IncoreGDF"], default="GDF", help="Density-fitting method")
    parser.add_argument("--auxbasis", help="Auxiliary basis. Only works for those known to PySCF.", default='auto')
    parser.add_argument("--auxbasis-file", help="Load auxiliary basis from file (NWChem format)")
    parser.add_argument("--save-gdf", help="Save primitive cell GDF") #, default="gdf-%.2f.h5")
    parser.add_argument("--load-gdf", help="Load primitive cell GDF")
    parser.add_argument("--df-lindep-method")
    parser.add_argument("--df-lindep-threshold", type=float)
    parser.add_argument("--df-lindep-always", action="store_true")
    # Embedded correlated calculation
    parser.add_argument("--iao-minao", default="minao", help="Minimial basis set for IAOs.")
    parser.add_argument("--solver", type=str_or_none, default="CCSD")
    parser.add_argument("--ccsd-diis-start-cycle", type=int)
    parser.add_argument("--opts", nargs="*", default=[])
    parser.add_argument("--plot-orbitals-crop-c", type=float, nargs=2)
    parser.add_argument("--pop-analysis", type=str)
    parser.add_argument("--check-surrounding", type=int)
    # Bath specific
    parser.add_argument("--dmet-threshold", type=float, default=1e-4, help="Threshold for DMET bath orbitals. Default= 1e-4")
    parser.add_argument("--bno-threshold", type=float, nargs="*",
            default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            help="Tolerance for additional bath orbitals. If positive, interpreted as an occupation number threshold.")
    #parser.add_argument("--mp2-correction", type=int, choices=[0, 1], default=1, help="Calculate MP2 correction to energy.")
    # Other
    parser.add_argument("--run-hf", type=int, default=1)
    parser.add_argument("--run-ewf", type=int, default=1)

    # Benchmark
    #parser.add_argument("--dft-xc", nargs="*", default=[])
    #parser.add_argument("--canonical-mp2", action="store_true", help="Perform canonical MP2 calculation.")
    #parser.add_argument("--canonical-ccsd", action="store_true", help="Perform canonical CCSD calculation.")
    parser.add_argument("--benchmarks", nargs='*')

    args, restargs = parser.parse_known_args()
    sys.argv[1:] = restargs

    # System specific default arguments
    if args.system == "diamond":
        defaults = {
                "atoms" : ["C", "C"],
                "ndim" : 3,
                #"lattice_consts" : np.arange(3.4, 3.8+1e-12, 0.1),
                # a0 ~ 3.56
                "lattice_consts" : np.arange(3.4, 3.7+1e-12, 0.05), # 7 calculations
                }
    elif args.system == "graphite":
        defaults = {
                "atoms" : ["C", "C", "C", "C"],
                "ndim" : 3,
                #"lattice_consts" : np.arange(2.35, 2.55+1e-12, 0.05),
                "lattice_consts" : np.asarray([2.4, 2.425, 2.45, 2.475, 2.5]),
                "vacuum_size" : 6.708,
                }
    elif args.system == "graphene":
        defaults = {
                "atoms" : ["C", "C"],
                "ndim" : 2,
                #"lattice_consts" : np.arange(2.35, 2.6+1e-12, 0.05),
                "lattice_consts" : np.asarray([2.4, 2.425, 2.45, 2.475, 2.5, 2.525]),
                "vacuum_size" : 20.0
                }

    elif args.system == "hbn":
        defaults = {
                "atoms" : ["B", "N"],
                "ndim" : 2,
                "lattice_consts" : np.asarray([2.45, 2.475, 2.5, 2.525, 2.55, 2.575]),
                "vacuum_size" : 20.0
                }

    elif args.system in ('perovskite', 'SrTiO3'):
        defaults = {
                "atoms" : ["Sr", "Ti", "O"],
                "ndim" : 3,
                #"lattice_consts" : np.arange(3.8, 4.0+1e-12, 0.05),
                #"lattice_consts" : np.arange(3.825, 3.975+1e-12, 0.025),
                "lattice_consts" : np.asarray([3.905])
                }
    elif args.system == "SrTiO3-I4":
        defaults = {
                "atoms" : ["Sr", "Ti", "O"],
                "ndim" : 3,
                #"lattice_consts" : np.arange(3.8, 4.0+1e-12, 0.05),
                #"lattice_consts" : np.arange(3.825, 3.975+1e-12, 0.025),
                #"lattice_consts" : np.asarray([3.905])
                #"lattice_consts" : np.asarray([5.507])
                "lattice_consts" : np.asarray([5.522])
                }

    if args.auxbasis is not None and args.auxbasis.lower() == 'auto':
        args.auxbasis = default_auxbasis.get(args.basis)
        log.warning("Setting auxbasis for basis %s to %s", args.basis, args.auxbasis)

    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    args.lattice_consts = np.asarray(args.lattice_consts)[args.skip:]
    if args.only is not None:
        args.lattice_consts = args.lattice_consts[args.only]

    if MPI_rank == 0:
        log.info("PARAMETERS IN INPUT SCRIPT")
        log.info("**************************")
        for name, value in sorted(vars(args).items()):
            log.info("  > %-32s %r", name+':', value)

    return args


def make_cell(a, args, **kwargs):

    cell = pyscf.pbc.gto.Cell()
    if args.system == "diamond":
        amat, atom = molstructs.diamond(args.atoms, a=a)
    elif args.system == "graphite":
        amat, atom = molstructs.graphite(args.atoms, a=a, c=args.vacuum_size)
    elif args.system in ("graphene", "hbn"):
        amat, atom = molstructs.graphene(args.atoms, a=a, c=args.vacuum_size)
    elif args.system in ("perovskite", 'SrTiO3'):
        amat, atom = molstructs.perovskite(args.atoms, a=a)
    elif args.system == "SrTiO3-I4":
        amat, atom = molstructs.perovskite_tetragonal(args.atoms)
        # Cubic structure:
        #a0 = 3.905
        #amat, atom = molstructs.perovskite_tetragonal(args.atoms, a=np.sqrt(2)*a0, c=2*a0, u=0.25)

    cell.a, cell.atom = amat, atom
    cell.dimension = args.ndim
    cell.precision = args.precision
    cell.verbose = args.pyscf_verbose
    cell.basis = kwargs.get("basis", args.basis)
    if args.pseudopot:
        cell.pseudo = args.pseudopot
    if args.ecp:
        cell.ecp = args.ecp
    if args.exp_to_discard:
        cell.exp_to_discard = args.exp_to_discard
    if args.lindep_threshold:
        cell.lindep_threshold=args.lindep_threshold

    cell.build()
    if args.supercell and not np.all(args.supercell == 1):
        cell = pyscf.pbc.tools.super_cell(cell, args.supercell)

    # Counterpoise
    if args.counterpoise is not None:
        # Remove all but one atom
        if '*' not in args.counterpoise:
            atmidx = int(args.counterpoise)
            cell = counterpoise.make_mol(cell, atmidx, rmax=0, output='pyscf-cp-m.txt')
        # Remove all but one atom, but keep basis functions of the 26 (3D) or 8 (2D) neighboring cells
        elif '*' in args.counterpoise:
            atmidx = int(args.counterpoise.replace('*', ''))
            cell = counterpoise.make_mol(cell, atmidx, rmax=args.counterpoise_rmax, nimages=args.counterpoise_nimages,
                    unit=args.counterpoise_unit, output='pyscf-cp-d.txt')
    return cell


def pop_analysis(mf, filename=None, mode="a"):
    t0 = timer()
    if filename is None: filename = "popfile.txt"
    lo = pyscf.lo.orth_ao(mf.mol, "lowdin")
    mo = np.linalg.solve(lo, mf.mo_coeff)
    dm = mf.make_rdm1(mo, mf.mo_occ)
    pop, chg = mf.mulliken_pop(dm=dm, s=np.eye(dm.shape[-1]))

    if filename:
        tstamp = datetime.now()
        log.info("[%s] Writing mean-field population analysis to file \"%s\"", tstamp, filename)
        with open(filename, mode) as f:
            f.write("[%s] Mean-field population analysis\n" % tstamp)
            f.write("*%s********************************\n" % (26*"*"))
            # per orbital
            for i, s in enumerate(mf.mol.ao_labels()):
                f.write("  * MF population of OrthAO %4d %-16s = %10.5f\n" % (i, s, pop[i]))
            # per atom
            f.write("[%s] Mean-field atomic charges\n" % tstamp)
            f.write("*%s***************************\n" % (26*"*"))
            for ia in range(mf.mol.natm):
                symb = mf.mol.atom_symbol(ia)
                f.write("  * MF charge at atom %3d %-3s = %10.5f\n" % (ia, symb, chg[ia]))
    log.info("Time for population analysis= %.2f s", (timer()-t0))

    return pop, chg

def get_mf(cell, kpts=None, xc='hf'):
    if kpts is None:
        if hasattr(cell, 'a') and cell.a is not None:
            if xc is None or xc.lower() == "hf":
                mf = pyscf.pbc.scf.RHF(cell)
            else:
                mf = pyscf.pbc.dft.RKS(cell)
                mf.xc = xc
        else:
            if xc is None or xc.lower() == "hf":
                mf = pyscf.scf.RHF(cell)
            else:
                mf = pyscf.dft.RKS(cell)
                mf.xc = xc
    else:
        if xc is None or xc.lower() == "hf":
            mf = pyscf.pbc.scf.KRHF(cell, kpts)
        else:
            mf = pyscf.pbc.dft.KRKS(cell, kpts)
            mf.xc = xc
    return mf

def run_mf(a, cell, args, kpts=None, dm_init=None, xc="hf", df=None, build_df_early=False):
    mf = get_mf(cell, kpts, xc)
    if args.exxdiv_none:
        mf.exxdiv = None
    if args.scf_conv_tol is not None:
        mf.conv_tol = args.scf_conv_tol
    if args.scf_max_cycle is not None:
        mf.max_cycle = args.scf_max_cycle
    # Load SCF from checkpoint file
    load_scf_ok = False
    if args.load_scf:
        fname = (args.load_scf % a) if ("%" in args.load_scf) else args.load_scf
        log.info("Loading SCF from file %s...", fname)
        try:
            chkfile_dict = pyscf.pbc.scf.chkfile.load(fname, "scf")
            log.info("Loaded attributes: %r", list(chkfile_dict.keys()))
            mf.__dict__.update(chkfile_dict)
            mf.converged = True
        except IOError:
            log.error("Could not load SCF from file %s. File not found." % fname)
            log.error("Calculating SCF instead.")
        except Exception as e:
            log.error("ERROR loading SCF from file %s", fname)
            log.error("Exception: %s", e)
            log.error("Calculating SCF instead.")
        else:
            load_scf_ok = True
            log.info("SCF loaded successfully.")

    #if args.lindep_threshold:
    #    log.debug("Adding remove_linear_dep_ to mf object with threshold= %.2e", args.lindep_threshold)
    #    mf = pyscf.scf.addons.remove_linear_dep_(mf, threshold=args.lindep_threshold)

    # Density-fitting
    if df is not None:
        mf.with_df = df
    #elif args.df == "GDF":
    elif args.df in ("GDF", "IncoreGDF"):
        if args.df == "GDF":
            mf = mf.density_fit()
        elif args.df == "IncoreGDF":
            mf.with_df = IncoreGDF(cell, kpts)
        df = mf.with_df
        # TEST
        if args.df_lindep_method is not None:
            df.linear_dep_method = args.df_lindep_method
        if args.df_lindep_threshold is not None:
            df.linear_dep_threshold = args.df_lindep_threshold
        # Always remove linear-dependency (do not try CD)
        if args.df_lindep_always:
            df.linear_dep_always = args.df_lindep_always

        if args.auxbasis is not None:
            log.info("Loading auxbasis %s.", args.auxbasis)
            df.auxbasis = args.auxbasis
        elif args.auxbasis_file is not None:
            log.info("Loading auxbasis from file %s.", args.auxbasis_file)
            df.auxbasis = {atom : pyscf.gto.load(args.auxbasis, atom) for atom in args.atoms}
        load_gdf_ok = False
        # Load GDF
        if args.load_gdf is not None:
            fname = (args.load_gdf % a) if ("%" in args.load_gdf) else args.load_gdf
            log.info("Loading GDF from file %s...", fname)
            if os.path.isfile(fname):
                df._cderi = fname
                load_gdf_ok = True
            else:
                log.error("Could not load GDF from file %s. File not found." % fname)
        # Calculate GDF
        if not load_gdf_ok:
            if args.save_gdf is not None:
                fname = (args.save_gdf % a) if ("%" in args.save_gdf) else args.save_gdf
                df._cderi_to_save = fname
                log.info("Saving GDF to file %s...", fname)
            if build_df_early:
                log.info("Building GDF...")
                t0 = timer()
                df.build()
                log.info("Time for GDF build: %.3f s", (timer()-t0))

    # Calculate SCF
    if not load_scf_ok:
        if args.save_scf:
            fname = (args.save_scf % a) if ("%" in args.save_scf) else args.save_scf
            mf.chkfile = fname

        t0 = timer()
        mf.kernel(dm0=dm_init)
        log.info("Time for MF: %.3f s", (timer()-t0))
    log.info("MF converged= %r", mf.converged)
    log.info("E(MF)= %+16.8f Ha", mf.e_tot)
    if not mf.converged:
        log.warning("WARNING: mean-field not converged!!!")

    # Check orthogonality
    if hasattr(mf, "kpts"):
        sk = mf.get_ovlp()
        for ik, k in enumerate(mf.kpts):
            c = mf.mo_coeff[ik]
            csc = np.linalg.multi_dot((c.T.conj(), sk[ik], c))
            err = abs(csc-np.eye(c.shape[-1])).max()
            if err > 1e-6:
                log.error("MOs not orthogonal at k-point %d. Error= %.2e", ik, err)
    else:
        s = mf.get_ovlp()
        c = mf.mo_coeff
        csc = np.linalg.multi_dot((c.T.conj(), s, c))
        err = abs(csc-np.eye(c.shape[-1])).max()
        if err > 1e-6:
            log.error("MOs not orthogonal. Error= %.2e", err)

    return mf

def run_benchmarks(a, cell, mf, kpts, args):
    energies = {}

    # DFT
    df = None
    #for xc in args.dft_xc:
    for bm in args.benchmarks:
        bm = bm.lower()
        if not bm.startswith('dft-'):
            continue
        xc = bm[4:]
        log.info("Running DFT xc= %s", xc)
        dft = run_mf(a, cell, args, kpts=kpts, xc=xc, df=df)
        if df is None:
            df = dft.with_df
        energies[xc] = [dft.e_tot]

        # population analysis
        if args.pop_analysis:
            dft_sc = pyscf.pbc.tools.k2gamma.k2gamma(dft)
            c_loc = pyscf.lo.orth_ao(dft_sc, 'lowdin')
            mo = np.linalg.solve(c_loc, dft_sc.mo_coeff)
            # Mulliken population analysis based on Lowdin orbitals
            dm = dft_sc.make_rdm1(mo, dft_sc.mo_occ)
            pop, chg = dft_sc.mulliken_pop(dm=dm, s=np.eye(dm.shape[-1]))
            with open('dft-%s-pop.txt' % xc, 'a') as f:
                for i in range(dft_sc.mol.natm):
                    f.write('%3d %6s  %.8f\n' % (i, dft_sc.mol.atom_symbol(i), chg[i]))


    # Canonical MP2
    if 'MP2' in args.benchmarks:
        try:
            t0 = timer()
            if hasattr(mf, "kpts"):
                raise NotImplementedError()
            else:
                mp2 = pyscf.pbc.mp.MP2(mf)
            mp2.kernel()
            log.info("Ecorr(MP2)= %.8g", mp2.e_corr)
            log.info("Time for canonical MP2: %.3f s", (timer()-t0))
            energies["mp2"] = [mp2.e_tot]
        except Exception as e:
            log.error("Error in canonical MP2 calculation: %s", e)

    # Canonical CCSD
    if 'CCSD' in args.benchmarks:
        try:
            t0 = timer()
            if hasattr(mf, "kpts"):
                cc = pyscf.pbc.cc.KCCSD(mf)
            else:
                raise NotImplementedError()
            cc.kernel()
            log.info("Canonical CCSD: E(corr)= %+16.8f Ha", cc.e_corr)
            log.info("Time for canonical CCSD: %.3f s", (timer()-t0))
            energies["cc"] = [cc.e_tot]
        except Exception as e:
            log.error("Error in canonical CCSD calculation: %s", e)
    return energies


args = get_arguments()

# Loop over geometries
for i, a in enumerate(args.lattice_consts):
    t0 = timer()

    if MPI_rank == 0:
        log.info("LATTICE CONSTANT %.3f", a)
        log.info("**********************")
        log.changeIndentLevel(1)

    energies = {}

    # Setup cell
    cell = make_cell(a, args)

    # k-points
    if args.k_points is None or np.product(args.k_points) == 1:
        kpts = None
    else:
        kpts = cell.make_kpts(args.k_points)

    if args.hf_init_guess_basis is not None:
        cell_init_guess = make_cell(a, args, basis=args.hf_init_guess_basis)
    else:
        cell_init_guess = cell

    if args.load_scf_init_guess:
        fname = (args.load_scf_init_guess % a) if ("%" in args.load_scf_init_guess) else args.load_scf_init_guess
        log.info("Loading initial guess for SCF from file %s...", fname)
        chkfile_dict = pyscf.pbc.scf.chkfile.load(fname, "scf")
        log.info("Loaded attributes: %r", list(chkfile_dict.keys()))
        occ0, c0 = chkfile_dict["mo_occ"], chkfile_dict["mo_coeff"]
        c0 = c0[:,occ0>0]
        c_init = c0
        dm_init = np.dot(c0 * occ0[occ0>0], c0.T.conj())
    elif args.hf_init_guess_basis is not None:
        log.info("Running initial guess HF")
        mf_init_guess = run_mf(a, cell_init_guess, args)
        dm_init = mf_init_guess.make_rdm1()
    else:
        dm_init = None

    # Convert DM to cell AOs
    if dm_init is not None and args.hf_init_guess_basis is not None:
        log.debug("Converting basis of initial guess DM from %s to %s", cell_init_guess.basis, cell.basis)
        dm_init = pyscf.pbc.scf.addons.project_dm_nr2nr(cell_init_guess, dm_init, cell, kpts)

    # TEST
    if args.check_surrounding is not None:
        scell = pyscf.pbc.tools.super_cell(cell, args.k_points)
        atom = args.check_surrounding
        center = scell.atom_coord(atom, unit='ANG')
        distances = np.linalg.norm(center[None] -  scell.atom_coords(unit='ANG'), axis=1)
        sort = np.argsort(distances)
        for atm in sort:
            print('%3d %-6s at %.6f %.6f %.6f  d= %.8f A' % (atm, scell.atom_symbol(atm),
                *scell.atom_coord(atm, unit='ANG'), distances[atm]))
        raise SystemExit()

    # Mean-field
    if args.run_hf:
        mf = run_mf(a, cell, args, kpts=kpts, dm_init=dm_init)

        with open("mo-energies.txt", "a") as f:
            if not isinstance(mf.mo_energy, list):
                np.savetxt(f, mf.mo_energy, fmt="%.10e", header="MO energies at a=%.2f" % a)
            else:
                for k, mok in enumerate(mf.mo_energy):
                    np.savetxt(f, mok, fmt="%.10e", header="MO energies at a=%.2f kpt= %d" % (a, k))

        #    np.savetxt(f, np.asarray(mf.mo_energy).T, fmt="%.10e", header="MO energies at a=%.2f" % a)
        energies["hf"] = [mf.e_tot]
    else:
        mf = None

    if args.supercell is not None:
        ncells = np.product(args.supercell)
    else:
        ncells = 1

    #elif args.k_points is not None:
    #    nkpts = np.product(args.k_points)
    #else:
    #    nkpts = 1

    # DFT and Post-HF benchmarks
    if args.benchmarks:
        energies.update(run_benchmarks(a, cell, mf, kpts, args))

    if args.run_ewf:
        energies["ccsd"] = []
        energies["ccsd-dmp2"] = []
        if args.solver == "CCSD(T)":
            energies["ccsdt"] = []
            energies["ccsdt-dmp2"] = []
        # Embedding calculations
        # ----------------------

        kwargs = {opt : True for opt in args.opts}
        if args.pop_analysis:
            kwargs['pop_analysis'] = args.pop_analysis
        solver_options = {}
        if args.ccsd_diis_start_cycle is not None:
            solver_options["diis_start_cycle"] = args.ccsd_diis_start_cycle
        ccx = vayesta.ewf.EWF(mf, solver=args.solver, iao_minao=args.iao_minao, dmet_threshold=args.dmet_threshold,
            bno_threshold=args.bno_threshold, solver_options=solver_options,
            **kwargs)

        # Define atomic fragments, first argument is atom index
        if args.counterpoise:
            idx = int(args.counterpoise.replace('*', ''))
            ccx.make_atom_fragment(idx, **kwargs)
        else:
            if args.system == "diamond":
                ccx.make_atom_fragment(0, sym_factor=2*ncells)
            elif args.system == "graphite":
                ccx.make_atom_fragment(0, sym_factor=2*ncells)
                ccx.make_atom_fragment(1, sym_factor=2*ncells)
            elif args.system in ("graphene", "hbn"):
                #for ix in range(2):
                #    ccx.make_atom_cluster(ix, sym_factor=ncells, **kwargs)
                #if ncells % 2 == 0:
                #    nx, ny = args.supercell[:2]
                #    ix = 2*np.arange(ncells).reshape(nx,ny)[nx//2,ny//2]
                #else:
                #    ix = ncells-1    # Make cluster in center
                #ccx.make_atom_cluster(ix, sym_factor=2, **kwargs)

                if (args.atoms[0] == args.atoms[1]):
                    ccx.make_atom_fragment(0, sym_factor=2*ncells, **kwargs)
                else:
                    ccx.make_atom_fragment(0, sym_factor=ncells, **kwargs)
                    ccx.make_atom_fragment(1, sym_factor=ncells, **kwargs)

            #elif args.system == "perovskite":
            #    # Ti needs larger threshold
            #    ccx.make_atom_fragment(0, sym_factor=ncells, bno_threshold_factor=0.3)
            #    ccx.make_atom_fragment(1, sym_factor=ncells)
            #    ccx.make_atom_fragment(2, sym_factor=3*ncells, bno_threshold_factor=0.03)
            elif args.system in ("perovskite", 'SrTiO3-I4'):
                ccx.make_atom_fragment(1, aos=['4s', '3d'], sym_factor=ncells)
            #elif args.system in ('perovskite',):
            #    # 8 fragment orbitals:
            #    aos = ['1 Ti 3dz', '1 Ti 3dx2-y2', '2 O 2px', '3 O 2py', '4 O 2pz', '22 O 2px', '13 O 2py', '9 O 2pz']
            #    ccx.make_ao_fragment(aos, sym_factor=ncells)
            #elif (args.system == 'SrTiO3-I4'):
	    #    #  6 Ti     at 2.761252 2.761252 3.905000  d= 0.00000000 A
	    #    # 17 O      at 2.761252 2.761252 1.952500  d= 1.95250000 A	-z
	    #    #  4 O      at 4.141878 1.380626 3.905000  d= 1.95250000 A	# Problem: x-y rotated!
	    #    # 19 O      at 1.380626 1.380626 3.905000  d= 1.95250000 A
	    #    # 94 O      at 4.141878 4.141878 3.905000  d= 1.95250000 A
	    #    # 49 O      at 1.380626 4.141878 3.905000  d= 1.95250000 A
	    #    #  7 O      at 2.761252 2.761252 5.857500  d= 1.95250000 A	+z
            #    aos = ['6 Ti 3dz', '6 Ti 3dx2-y2', ' O 2px', '3 O 2py', '4 O 2pz', '22 O 2px', '13 O 2py', '9 O 2pz']
            #    ccx.make_ao_fragment(aos, sym_factor=ncells)
            elif args.system == 'SrTiO3':
                # for 2x2x2:
                f = ccx.make_atom_fragment([1,2,3,4,22,13,9], sym_factor=ncells)
                iaos = ['1 Ti 3dz', '1 Ti 3dx2-y2', '2 O 2px', '3 O 2py', '4 O 2pz', '22 O 2px', '13 O 2py', '9 O 2pz']
                # Match must start with substring:
                # for 3x3x3:
                #f = ccx.make_atom_fragment([1,2,3,4,47,18,9], sym_factor=ncells)
                #iaos = ['1 Ti 3dz', '1 Ti 3dx2-y2', '2 O 2px', '3 O 2py', '4 O 2pz', '47 O 2px', '18 O 2py', '9 O 2pz']
                iaos = [('^%s' % iao) for iao in iaos]
                f.set_cas(iaos=iaos)
            else:
                raise RuntimeError()

        ccx.kernel()

        energies["ewf-ccsd"] = ccx.get_energies()

        # Write cluster sizes to file
        #for x in ccx.fragments:
        #    fname = "cluster-%s-size.txt" % x.id_name
        #    val = x.n_active
        #    with open(fname, "a") as f:
        #        f.write(("%6.3f" + len(val)*"  %3d" + "\n") % (a, *val))

        # Save energies
        #energies["ccsd-dmp2"].append((ccx.e_tot + ccx.e_delta_mp2))
        #if args.solver == "CCSD(T)":
        #    energies["ccsdt"].append((ccx.e_tot + ccx.e_pert_t))
        #    energies["ccsdt-dmp2"].append((ccx.e_tot + ccx.e_delta_mp2 + ccx.e_pert_t))

        del ccx

    if args.run_hf: del mf
    del cell

    # Write energies to files
    if MPI_rank == 0:
        for key, val in energies.items():
            if len(val) > 0:
                fname = "%s.txt" % key
                log.info("Writing to file %s", fname)
                with open(fname, "a") as f:
                    f.write(("%6.3f" + len(val)*"  %+16.8f" + "\n") % (a, *val))

    log.info("Total time for lattice constant %.2f= %.3f s", a, (timer()-t0))
    log.changeIndentLevel(-1)
