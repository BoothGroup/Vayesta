
atom_colors = dict(
    H="white",
    Li="darkgreen",
    B="pink",
    C="black",
    N="blue",
    O="red",
    F="cyan",
    Na="violet",
    Cl="green",
    Mg="darkgreen")

def get_atom_color(symbol, default=None):
    sym = ''.join([l for l in symbol if l.isalpha()])
    return atom_colors.get(sym, default)
