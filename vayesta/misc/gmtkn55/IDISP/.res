n=$1 
#DLPNO-CCSD(T) TCutPairs 10^-5 Eh CBS(aug-cc-pVTZ/aug-cc-pVQZ)
#(estimated uncertainty: +/- 0.25 kcal/mol)
tmer2_GMTKN antdimer ant      x  1 -2   $n -9.15 
tmer2_GMTKN pxylene pc22 h2 x  2 -1 -2  $n -60.28
tmer2_GMTKN octane1 octane2    x  1 -1  $n -1.21 
tmer2_GMTKN undecan1 undecan2  x  1 -1  $n  9.10 
tmer2_GMTKN F14f F14l x -1 1            $n -3.64
tmer2_GMTKN F22f F22l x -1 1            $n -1.96
