import jax
import jax.numpy as jnp


x = 2.
y = lambda z : z**3

grad = jax.grad(y)
grad2 = jax.grad(grad)

first = grad(x)
second= grad2(x)

print(f'First derivative of f({x}):', first)
print(f'Second derivative of f({x}):', second)


x = 3.
y = 3.

def fun(x, y):
    return 5*x**4 - 5*y*x**2 + 3*y**3 - 55*y + x*y - 3

grad = jax.grad(fun, argnums=(0, 1))

grad2 = jax.jacobian(grad, argnums=[0, 1])

first = grad(x, y)
second = grad2(x, y)

second_x = second[0][0]
second_y = second[1][1]

print(f'First derivative of f({x}, {y}):', first[0], first[1])
print(f'Second derivative of f({x}, {y}):', second_x, second_y)


def fun(x):
    return 3*x**3

A = jnp.array([3.0, 4.0, 5.0])

grad = jax.jacfwd(fun, argnums=(0))(A).sum(axis=1)
print(grad)



import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

x = jnp.expand_dims(jnp.linspace(-1, 1, 20), axis=1)

u = lambda x: jnp.sin(jnp.pi * x)
ux = jax.vmap(jax.vmap(jax.grad(u)))
uxx = jax.jacfwd(ux)(x).sum(axis=1).sum(axis=1).sum(axis=1)


plt.plot(x, u(x))
plt.plot(x, ux(x))
plt.plot(x, uxx)
plt.show()