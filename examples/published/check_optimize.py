"""This is an example script to verify the efficacy of the library's optimizer on the Beale function.
Related tests also implement the same functionality."""
import pygrank as pg

# Beale function
p = pg.optimize(loss=lambda p: (p[0] - 2) ** 2 + (p[1] - 1) ** 4, max_vals=[5, 5], parameter_tol=1.E-8)
print(p)
assert abs(p[0] - 2) < 1.E-6
assert abs(p[1] - 1) < 1.E-6
