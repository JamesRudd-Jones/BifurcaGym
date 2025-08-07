# from phi.jax.flow import *
import phi.jax.flow as phijax
from tqdm import trange
import matplotlib.pyplot as plt
phijax.math.set_global_precision(64)


DOMAIN = dict(extrapolation=phijax.extrapolation.PERIODIC, bounds=phijax.Box(x=2*phijax.PI, y=2*phijax.PI), x=100, y=100)
FORCING = phijax.CenteredGrid(lambda x, y: phijax.vec(x=phijax.math.sin(4 * y), y=0), **DOMAIN) + phijax.CenteredGrid(phijax.Noise(), **DOMAIN) * 0.01
phijax.plot({'Force along X': FORCING['x'], 'Force along Y': FORCING['y']}, same_scale=False)
plt.show()

def momentum_equation(v, viscosity=0.001):
    advection = phijax.advect.finite_difference(v, v, order=6)
    diffusion = phijax.diffuse.finite_difference(v, viscosity, order=6)
    return advection + diffusion + FORCING

@phijax.jit_compile
def rk4_step(v, p, dt):
  return phijax.fluid.incompressible_rk4(momentum_equation, v, p, dt, pressure_order=4, pressure_solve=phijax.Solve('CG', 1e-5, 1e-5))


v0 = phijax.CenteredGrid(phijax.tensor([0, 0], phijax.channel(vector='x, y')), **DOMAIN)
p0 = phijax.CenteredGrid(0, **DOMAIN)
multi_step = lambda *x, **kwargs: phijax.iterate(rk4_step, 25, *x, **kwargs)
v_trj, p_trj = phijax.iterate(multi_step, phijax.batch(time=100), v0, p0, dt=0.005, range=trange)
phijax.vis.plot(phijax.field.curl(v_trj.with_extrapolation(0)), animate='time')

