

def log_orbitals(logger, labels, ncol=10):
    # Group orbitals by atom - list(dict.fromkeys(...)) to only get unique atom indices:
    atoms = list(dict.fromkeys([l[0] for l in labels]))
    labels = [[l for l in labels if (l[0] == atom)] for atom in atoms]
    for atom in labels:
        prefix = "Atom %4s %3s:" % (atom[0][0], atom[0][1])
        # Print up to ncol orbitals per line:
        for idx in range(0, len(atom), ncol):
            line = atom[idx:idx+ncol]
            fmt = '  %13s ' + len(line)*' %6s'
            # Preformat
            orbs = [('%s-%s' % (nl, ml) if ml else nl) for a, sym, nl, ml in line]
            logger(fmt, prefix, *orbs)
            prefix = ""
