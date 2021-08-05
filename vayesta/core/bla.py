import numpy as np
import scipy
import scipy.special
import scipy.integrate

from matplotlib import pyplot as plt


def func(r, omega=1, rcut=10):
    return scipy.special.erf(omega*r)/r #* np.heaviside(r-rcut, 0.5)


maxval = 100
num = 1000
grid = np.linspace(-maxval, maxval, num)

y = func(grid)

plt.plot(grid, y)
plt.show()


ff = lambda r, k : func(r) * np.exp(1j*r*k)

k = 0
y = ff(grid, k=k)
plt.plot(grid, y.real)
plt.plot(grid, y.imag)
plt.show()


res = scipy.integrate.quad(ff, -maxval, maxval, args=k)
print(res)
