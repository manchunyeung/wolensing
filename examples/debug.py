import numpy as np
import jax.numpy as jnp

x1s = np.loadtxt('./x1s.txt')
x2s = np.loadtxt('./x2s.txt')

x1, x2 = np.meshgrid(x1s, x2s)

print(x1.dtype)

x1, x2 = x1[:4, :4], x2[:4, :4]
# x1 = np.loadtxt('./x1.txt')[:5, :5]
# x2 = np.loadtxt('./x2.txt')[:5, :5]
y = np.loadtxt('./y.txt')

# print(x1, '\n', x2, '\n', y)

def j_geometrical(x1, x2, y):
    x = jnp.array([x1, x2], dtype=np.float64)
    # print(x[0], '\n', x[1])
    return (1/2) * jnp.linalg.norm(x-y[:, jnp.newaxis, jnp.newaxis], axis=0)**2

def geometrical(x1, x2, y):
    return (1/2) * ((x1-y[0])**2 + (x2-y[1])**2)



geo = geometrical(x1, x2, y)
geo1 = j_geometrical(x1, x2, y)
print(geo, '\n\n', geo1)
# np.savetxt('./geo2.txt', geo)


