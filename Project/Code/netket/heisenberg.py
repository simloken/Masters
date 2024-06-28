import netket as nk
import os
from scipy.optimize import curve_fit
import numpy as np
def approximate_gs_heisenberg(arr):
    vals = []
    
    for i in arr:
        print(i)
    
        g = nk.graph.Hypercube(length=int(i), n_dim=1, pbc=True)
        hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
        ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
        evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
        exact_gs_energy = evals[0]
        vals.append(exact_gs_energy)
    
    def cubic_function(L, a, b, c, d):
        return a * L**3 + b * L**2 + c * L + d

    params, covariance = curve_fit(cubic_function, arr, vals)

    a, b, c, d = params
    
    file_path = os.path.join("..", "Data", "heisenberg_approx.txt")

    with open(file_path, "w") as f:
        f.write(f"a: {a}\n")
        f.write(f"b: {b}\n")
        f.write(f"c: {c}\n")
        f.write(f"d: {d}\n")
        
        
        
arr = np.array([2, 4, 6, 8, 10, 12,
                14, 16, 18, 20, 22, 24]) #length of array is dependent on available memory, I cannot go higher than 24
approximate_gs_heisenberg(arr)