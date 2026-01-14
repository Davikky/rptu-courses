# RPTU-course: Introduction to Scientic Computing (JC - Model)

The Hamiltonian describing this system is:

$$
H = \frac{1}{2}\omega_{q}\sigma_{z} + \omega_{0}a^{\dagger}a
    + g(\sigma_{-}a^{\dagger}\, +\, \sigma_{+}a).
$$

---
### A. Basis Ordering and Labelling

We begin by specifying the following canonical order of our vector space for N-bosonic modes:

$$ (e_0, e_1 ,…,e_{N−1}, g_0,\, g_1,…,g_{N−1}) $$

with:

- total dimension = $2N$
- $g_0$ is the $(N+1)th$ component → this fixes the ordering convention
that is,

- first half = excited qubit states
- second half = ground qubit states

This matches QuTiP tensor ordering

`So our construction is:` ***tensor(qubit, boson)***

---
### Label Generator
We begin by constructing our label generator so that it is robust and reusable

```
# label generator of basis
def basis_labels(N):
    labels = []
    for n in range(N):
        labels.append(f"e{n}")
    for n in range(N):
        labels.append(f"g{n}")
    return labels
```
> Example for `N=5`:  
  labels = ['e0','e1','e2','e3','e4','g0','g1','g2','g3','g4']

---
## B. Jaynes-Cummings model

Here, we implement the JC Hamiltonian to:
- vary g
- restrict ω₀ to a fixed range
- return eigenvalues and eigenstates

The Hamiltonian describing this system is:

$$
\frac{H}{\hbar} = \frac{1}{2}\omega_{q}\sigma_{z} + \omega_{0}a^{\dagger}a
    + g(\sigma_{-}a^{\dagger}\, +\, \sigma_{+}a).
$$

> `Note: In our implementation, we have chosen to set` $w_q = 1$

```
# install qutip (if necessary)
#!pip install qutip 

# import useful libraries
from qutip import *
import numpy as np

# hamiltonian design: JC-model
def jc_model(N, g, omega_0):
    '''
    This function implements the JC-Hamiltonian with:
        N = number of bosonic states
        g = coupling constant
        omega_0 = boson frequency
        >>> "see eqn above"
    return eigenvalues and eigenstates
    '''

    a  = tensor(identity(2), destroy(N))
    sm = tensor(0.5*(sigmax() - 1j*sigmay()), identity(N))
    sz = tensor(sigmaz(), identity(N))

    H = 0.5 * sz + omega_0 * a.dag() * a + g * (sm * a.dag() + sm.dag() * a)
    return H.eigenstates()

```
### Occupation extraction

Our choice of design here is to implement:
- arbitrary energy level index
- arbitrary g
- a sweep over ω₀

```
# occupation extraction (key data structure)
def compute_occupation(N, g, omega_list, energy_index):
    '''
    This function generates a list of occupation amplitudes
    as a list of lists.

    return occupation
        - occupation[i] is the list of coefficients of choice energy_index for basis_component i
            (e.g. i=1 gives amps. of e1, and i=N/2 gives amps. of g0, for various values of ω₀)
        - occupation[i][k] is the coefficient of component i at omega_list[k]
    '''
    dim = 2 * N
    occupation = [[] for _ in range(dim)]

    for omega_0 in omega_list:
        energies, states = jc_model(N, g, omega_0)
        psi = states[energy_index]
        coeffs = psi.full().flatten()

        for i in range(dim):
            occupation[i].append(coeffs[i])

    return occupation
```
---
### Plotting with physical labels

Next we implement plots for:
- selected components (e.g. g0, e1, e3, g1,...)
- occupation probabilities (amp. square values)
- selected energy level (E0, E1, E2...)
- selected g

```
# Plotting with physical labels

import matplotlib.pyplot as plt

def plot_components(
    N,
    g,
    energy_index,
    selected_components,
    omega_list
):
    labels = basis_labels(N)
    occupation = compute_occupation(N, g, omega_list, energy_index)

    plt.figure(figsize=(7,5))

    for i in selected_components:
        y = np.abs(occupation[i])**2
        plt.plot(omega_list, y, label=labels[i])

    plt.xlabel(r'$\omega_0$')
    plt.ylabel(r'$|c_i|^2$')
    plt.title(f'Energy level {energy_index}, g = {g}')
    plt.legend()
    plt.show()
```
---
## Interactive widgets

```
# import useful libraries
import ipywidgets as widgets
from IPython.display import display

# Fixed ω₀ range
omega_list = np.linspace(0.1, 3.0, 80)

# Choose N value
'''??? Enter value of N below ???'''
N = 8 #default choice
```
### widget 1: Coupling strength g

```
g_slider = widgets.FloatSlider(
    value=0,
    min=0.0,
    max=1.0,
    step=0.02,
    description='g'
)
```
### widget 2: Energy level selector
```
# energy level selector
energy_selector = widgets.IntSlider(
    value=0,
    min=0,
    max=2*N - 1,
    step=1,
    description='Energy idx'
)
```
### widget 3: Component selector
```
# components with physical label
labels = basis_labels(N)

component_selector = widgets.SelectMultiple(
    options=[(labels[i], i) for i in range(2*N)],
    value=(0, N),   # e0 and g0 by default
    description='Components',
    rows=7
)
```
```
# Interactive binding
widgets.interactive(
    plot_components,
    N=widgets.fixed(N),
    g=g_slider,
    energy_index=energy_selector,
    selected_components=component_selector,
    omega_list=widgets.fixed(omega_list)
)
```
---
RPTU, Kaiserslautern
(c) Jan, 2026
