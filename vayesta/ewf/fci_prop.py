try:
    import pygnme
except:
    raise ImportError

def get_fci_var_energy(emb, kwargs=...):
    """Get variational energy from non-orthogonal FCI solutions over all cluster pairs.
    Also return 1RDM, and H and S between (unprojected) cluster FCI solutions.
    """
    
    emb.require_complete_fragmentation("Energy will not be accurate.", incl_virtual=False)
    nocc, nvir = emb.nocc, emb.nvir
    # Preconstruct some matrices, since the construction scales as N^3
    ovlp = emb.get_ovlp()
    cs_occ = np.dot(emb.mo_coeff_occ.T, ovlp)
    cs_vir = np.dot(emb.mo_coeff_vir.T, ovlp)

    for fx in emb.get_fragments(active=True, mpi_rank=mpi.rank):

        # Check fx.results.wf is FCI wave function of cluster fx

        cx_occ = fx.get_overlap('mo[occ]|cluster[occ]')
        cx_vir = fx.get_overlap('mo[vir]|cluster[vir]')
        cfx = fx.get_overlap('cluster[occ]|frag')
        mfx = fx.get_overlap('mo[occ]|frag')

        for fy in emb.get_fragments(active=True):

            # Consider overlaps
            cy_frag = fy.c_frag
            cy_occ_ao = fy.cluster.c_occ
            cy_vir_ao = fy.cluster.c_vir

            cy_occ = np.dot(cs_occ, cy_occ_ao)
            cy_vir = np.dot(cs_vir, cy_vir_ao)
            # Overlap between cluster x and cluster y:
            rxy_occ = np.dot(cx_occ.T, cy_occ)
            rxy_vir = np.dot(cx_vir.T, cy_vir)
            mfy = np.dot(cs_occ, cy_frag)

            # TODO: If using MPI, we'll need to get these via RMA

            # Check fy.results.wf is FCI object
