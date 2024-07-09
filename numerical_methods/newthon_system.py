import jax.numpy as jnp
from jax import jacobian, jit
from jax.scipy.linalg import solve

from numerical_methods.jacobian import JacobianTrail


def newton_system(F, x0: jnp.ndarray, tol=1e-6, max_iter=100):
    J = jit(jacobian(F))
    trail = JacobianTrail(F, J)

    x = x0.copy()
    trail.record(x)

    for _ in range(1, max_iter):
        Fx = F(x)
        Jx = J(x)
        s = solve(Jx, -Fx)

        if jnp.max(jnp.abs(s)) < tol:
            break

        x += s
        trail.record(x)

    return trail