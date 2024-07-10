from typing import Callable

import jax.numpy as jnp
from jax import jacobian, jit

from numerical_methods.newton_system_trail import NewtonSystemTrail


def newton_system(F: Callable[[jnp.ndarray], jnp.ndarray], x0: jnp.ndarray, tol=1e-6, max_iter=100):
    J = jit(jacobian(F))
    trail = NewtonSystemTrail(F, J)

    x = x0.copy()
    trail.record(x)

    for _ in range(1, max_iter):
        s = jnp.linalg.solve(J(x), -F(x))

        if jnp.max(jnp.abs(s)) < tol:
            break

        x += s
        trail.record(x)

    return trail
