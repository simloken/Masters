import tensorflow as tf
import numpy as np

from wavefunctions import Wavefunctions

def trial_wavefunction(name):
    init = Wavefunctions(name)
    return init.wf

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
   
def pre_train_NN(model, name, num_particles, dof):
    lattice = ['ising', 'heisenberg']
    x = [tf.random.normal((100, dof), dtype=tf.float32) 
             for _ in range(num_particles)]
    
    x = tf.concat(x, axis=-1)
    
    if name in lattice:
        x = np.where(np.random.uniform(size=(1000, num_particles)) > 0.5, 0.5, -0.5)
    elif name == 'two_fermions':
        x = [tf.random.normal((100, dof), dtype=tf.float32) 
                 for _ in range(num_particles)]
        x = tf.concat(x, axis=-1)
        
    optimizer = tf.keras.optimizers.Adam()
    
    y_values = trial_wavefunction(name)(x)
    
    for epoch in range(500):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_fn(y_values, y_pred)
        gradients = tape.gradient(loss, model.NN.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.NN.model.trainable_variables))
            
    model.NN.model.save_weights(f'./weights/{name}_weights.h5')