import numpy as np
def load_weights(name):
    # Load the weights from the .npz file
    file_path = f'./weights/{name}_weights.npz'
    with np.load(file_path) as data:
        W = data['W']
        a = data['a']
        b = data['b']
    return W, a, b