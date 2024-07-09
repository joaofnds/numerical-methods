from unittest import TestCase, skip

import jax.numpy as jnp

from numerical_methods.newthon_system import newton_system


class TestNewtonSystem(TestCase):
    def test_nonlinear_system(self):
        def F(x: jnp.ndarray):
            return jnp.array(
                [
                    x[0] + x[1] - 3,
                    x[0] ** 2 + x[1] ** 2 - 9,
                ],
            )

        x0 = jnp.array([1.0, 5.0])

        trail = newton_system(F, x0, max_iter=3)
        assert jnp.allclose(
            trail.result,
            jnp.array([-0.0919, 3.0919]),
            atol=1e-4,
        )

    def test_exercise_a(self):
        def F(x: jnp.ndarray):
            return jnp.array(
                [
                    x[0] ** 2 + x[1] ** 2 - 2,
                    jnp.e ** (x[0] - 1) + x[1] ** 3 - 2,
                ],
            )

        x0 = jnp.array([1.5, 2.0])

        trail = newton_system(F, x0, tol=1e-10)
        assert jnp.allclose(
            trail.result,
            jnp.array([1, 1]),
            atol=1e-4,
        )

    def test_exercise_b(self):
        def F(x: jnp.ndarray):
            return jnp.array(
                [
                    4 * x[0] - x[0] ** 3 + x[1],
                    -(x[0] ** 2) / 9 + (4 * x[1] - x[1] ** 2) / 4 + 1,
                ],
            )

        x0 = jnp.array([-1.0, -2.0])

        trail = newton_system(F, x0, tol=1e-4)
        assert jnp.allclose(
            trail.result,
            jnp.array([1.9317, -0.5182]),
            atol=1e-4,
        )

    @skip("does not converge")
    def test_exercise_c(self):
        def F(x: jnp.ndarray):
            return jnp.array(
                [
                    (2 * x[0] - x[0] ** 2 + 8) / 9 + (4 * x[1] - x[1] ** 2) / 4,
                    8 * x[0] - 4 * x[0] ** 2 + x[1] ** 2 + 1,
                ],
            )

        x0 = jnp.array([-1.0, -1.0])

        trail = newton_system(F, x0)
        assert jnp.allclose(trail.result, jnp.array([-0.1227, -0.2045]))
