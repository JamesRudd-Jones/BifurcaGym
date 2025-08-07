import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
import seaborn as sns
import xarray
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import matplotlib.pyplot as plt
import jax_cfd.base as base
import jax_cfd.collocated as collocated
import jax_cfd.data.xarray_utils as xru


funcutils = base.funcutils


def run_collocated(size, seed=0, inner_steps=25, outer_steps=100):
  density = 1.
  viscosity = 1e-3
  max_velocity = 2.0
  cfl_safety_factor = 0.5

  # Define the physical dimensions of the simulation.
  grid = base.grids.Grid((size, size),
                         domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

  # Choose a time step.
  dt = base.equations.stable_time_step(
      max_velocity, cfl_safety_factor, viscosity, grid)

  # Construct a random initial velocity. The `filtered_velocity_field` function
  # ensures that the initial velocity is divergence free and it filters out
  # high frequency fluctuations.
  v0 = collocated.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(seed), grid, max_velocity)

  v0[0].array.data = jnp.zeros((size, size))

  # v0[1] = v0[1].replace(data=jnp.zeros((64, 64)))

  forcing = ""

  # Define a step function and use it to compute a trajectory.
  # For linear convection, add the argument to semi_implicit_navier_stokes:
  #   `convect=collocated.advection.convect_linear`
  step_fn = funcutils.repeated(collocated.equations.semi_implicit_navier_stokes(density=density,
                                                                                viscosity=viscosity,
                                                                                dt=dt,
                                                                                grid=grid,
                                                                                forcing=forcing),
                               steps=inner_steps)
  rollout_fn = jax.jit(funcutils.trajectory(step_fn, outer_steps))
  _, trajectory = jax.device_get(rollout_fn(v0))

  yessay = trajectory[0].data

  # load into xarray for visualization and analysis
  ds = xarray.Dataset(
      {
          'u': (('time', 'x', 'y'), trajectory[0].data),
          'v': (('time', 'x', 'y'), trajectory[1].data),
      },
      coords={
          'x': grid.axes()[0],
          'y': grid.axes()[1],
          'time': dt * inner_steps * np.arange(outer_steps)
      }
  )
  return ds

ds_collocated_256 = run_collocated(size=64)


def vorticity(ds):
  return (ds.v.differentiate('x')
          - ds.u.differentiate('y')).rename('vorticity')

ds_collocated_256.pipe(vorticity).thin(time=10).plot.imshow(col='time', cmap=sns.cm.icefire, robust=True, col_wrap=5)
plt.show()


# physical parameters
# viscosity = 1e-3
# max_velocity = 7
# grid = grids.Grid((64, 64), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
# dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)
#
# # setup step function using crank-nicolson runge-kutta order 4
# smooth = True # use anti-aliasing
#
#
# # **use predefined settings for Kolmogorov flow**
# step_fn = spectral.time_stepping.crank_nicolson_rk4(spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)
#
#
# # run the simulation up until time 25.0 but only save 10 frames for visualization
# final_time = 25.0
# outer_steps = 10
# inner_steps = (final_time // dt) // 10
#
# trajectory_fn = cfd.funcutils.trajectory(cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)
#
# # create an initial velocity field and compute the fft of the vorticity.
# # the spectral code assumes an fft'd vorticity for an initial state
# v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid, max_velocity, 4)
# vorticity0 = cfd.finite_differences.curl_2d(v0).data
# vorticity_hat0 = jnp.fft.rfftn(vorticity0)
#
# _, trajectory = trajectory_fn(vorticity_hat0)
#
# # transform the trajectory into real-space and wrap in xarray for plotting
# spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # same for x and y
# coords = {
#   'time': dt * jnp.arange(outer_steps) * inner_steps,
#   'x': spatial_coord,
#   'y': spatial_coord,
# }
# xarray.DataArray(
#     jnp.fft.irfftn(trajectory, axes=(1,2)),
#     dims=["time", "x", "y"], coords=coords).plot.imshow(
#         col='time', col_wrap=5, cmap=sns.cm.icefire, robust=True)
# plt.show()