from collections.abc import Callable
from dataclasses import dataclass


ROLES = ("suppress",         # stabilise an unstable fixed point or orbit embedded in the chaos
         "navigate",         # reach a goal through chaos the agent cannot change
         "swing-up",         # pass through a chaotic region to an unstable equilibrium
         "induce",           # increase chaos or mixing
         "synchronise",      # bring coupled oscillators into step, or apart
         "control-induced",  # the uncontrolled system settles, chaos comes from actuation or feedback
         "tipping",          # steer into the right basin of attraction, not necessarily chaotic
         "reference",        # the same kind of task without chaos, for controlled comparisons
         )
TIME_STRUCTURES = ("map",
                   "ODE",
                   "DDE",
                   "PDE",
                   )
ENERGY = ("dissipative",     # volume contracting, an attractor
          "conservative",    # volume preserving, Hamiltonian or incompressible, no attractor
          "non-invertible",  # 1D maps, folding rather than contracting
          "stochastic",      # noise driven
          )
FORCING = ("autonomous",
           "forced",
           )
HYPERBOLICITY = ("hyperbolic",           # uniformly hyperbolic (or uniformly expanding), linear response expected
                 "singular hyperbolic",  # hyperbolic apart from a singularity, e.g. Lorenz63 (Tucker 2002)
                 "non-hyperbolic",       # tangencies or critical points, linear response not guaranteed
                 "unknown",              # not established, e.g. turbulent flows
                 "n/a",                  # not chaotic
                 )


@dataclass(frozen=True)
class EnvMetadata:
    role: str
    time_structure: str
    energy: str
    forcing: str
    hyperbolicity: str
    chaotic_uncontrolled: bool | None  # chaos expected under zero control from the reset distribution, None if unknown
    note: str = ""


ENV_METADATA = {
    # # # Classical control
    ####################################################################################################################
    "Acrobot-v0": EnvMetadata("swing-up", "ODE", "conservative", "autonomous", "non-hyperbolic", False,
                              "frictionless double pendulum, chaotic only at energies well above the hanging reset"),
    "CartPole-v0": EnvMetadata("reference", "ODE", "dissipative", "autonomous", "n/a", False,
                               "single pole with cart friction, the non-chaotic counterpart of NCartPole"),
    "NCartPole-v0": EnvMetadata("swing-up", "ODE", "dissipative", "autonomous", "non-hyperbolic", False,
                                "cart friction only, chaotic at swing-up energies rather than at the hanging reset"),
    "NPendulum-v0": EnvMetadata("swing-up", "ODE", "conservative", "autonomous", "non-hyperbolic", True,
                                "frictionless, mixed phase space, some reset states (angles up to pi/2) are already "
                                "chaotic"),
    "Pendubot-v0": EnvMetadata("swing-up", "ODE", "conservative", "autonomous", "non-hyperbolic", False,
                               "as Acrobot with the base joint actuated"),
    "Pendulum-v0": EnvMetadata("reference", "ODE", "conservative", "autonomous", "n/a", False,
                               "single pendulum, integrable, the non-chaotic counterpart of NPendulum"),
    "WetChicken-v0": EnvMetadata("reference", "map", "stochastic", "autonomous", "n/a", False,
                                 "unpredictable from noise, not chaos"),



    # # # Continuous time chaos
    ####################################################################################################################
    "ABCFlow-v0": EnvMetadata("navigate", "ODE", "conservative", "autonomous", "non-hyperbolic", True,
                              "steady incompressible 3D flow, chaotic streamlines between regular vortex tubes"),
    "BickleyJetFlow-v0": EnvMetadata("navigate", "ODE", "conservative", "forced", "non-hyperbolic", True,
                                     "time dependent area preserving jet, chaotic mixing either side of a barrier"),
    "Chua-v0": EnvMetadata("suppress", "ODE", "dissipative", "autonomous", "non-hyperbolic", True,
                           "double scroll attractor"),
    "DoubleGyreFlow-v0": EnvMetadata("navigate", "ODE", "conservative", "forced", "non-hyperbolic", True,
                                     "time periodic area preserving flow"),
    "KS-v0": EnvMetadata("suppress", "PDE", "dissipative", "autonomous", "non-hyperbolic", True,
                         "spatiotemporal chaos, L = 22"),
    "Lorenz63-v0": EnvMetadata("suppress", "ODE", "dissipative", "autonomous", "singular hyperbolic", True,
                               "Tucker (2002)"),
    "Rossler-v0": EnvMetadata("suppress", "ODE", "dissipative", "autonomous", "non-hyperbolic", True,
                              "spiral attractor"),



    # # # Discrete time chaos
    ####################################################################################################################
    "HenonMap-v0": EnvMetadata("suppress", "map", "dissipative", "autonomous", "non-hyperbolic", True,
                               "close to a boundary crisis"),
    "IkedaMap-v0": EnvMetadata("suppress", "map", "dissipative", "autonomous", "non-hyperbolic", True),
    "LogisticMap-v0": EnvMetadata("suppress", "map", "non-invertible", "autonomous", "non-hyperbolic", True,
                                  "critical point at x = 1/2, period doubling route to chaos in r"),
    "TentMap-v0": EnvMetadata("suppress", "map", "non-invertible", "autonomous", "hyperbolic", True,
                              "uniformly expanding, |f'| = mu everywhere"),
    "TinkerbellMap-v0": EnvMetadata("suppress", "map", "dissipative", "autonomous", "non-hyperbolic", True,
                                    "c = 1.8, moved off the canonical crisis"),



    # # # Fluid control
    ####################################################################################################################
    "FluidicPinball-v0": EnvMetadata("suppress", "PDE", "dissipative", "autonomous", "unknown", None,
                                     "Re = 105 may be periodic vortex shedding rather than chaos, to be measured"),
    "HydroGym-Kolmogorov-v0": EnvMetadata("suppress", "PDE", "dissipative", "autonomous", "unknown", True,
                                          "2D turbulence under a steady body force (steady, so autonomous)"),
    "HydroGym-Channel-v0": EnvMetadata("suppress", "PDE", "dissipative", "autonomous", "unknown", True,
                                       "turbulent channel flow under a steady pressure gradient"),



    # # # Locomotion
    ####################################################################################################################
    "Hopper-v0": EnvMetadata("control-induced", "ODE", "dissipative", "autonomous", "n/a", False,
                             "settles when uncontrolled, chaotic under periodic forcing"),
    "Walker2d-v0": EnvMetadata("control-induced", "ODE", "dissipative", "autonomous", "n/a", False,
                               "settles when uncontrolled, chaotic under periodic forcing"),
}


@dataclass(frozen=True)
class BifurcationDefault:
    param: str  # params field to sweep
    low: float
    high: float
    observable: Callable  # state -> scalar
    observable_name: str
    record: str  # "samples" for maps, "peaks" (successive maxima) for flows
    n_transient: int
    n_samples: int
    n_lyapunov_steps: int  # flows need many steps, 5,000 at dt ~0.01 is only 50 time units


BIFURCATION_DEFAULTS = {
    "LogisticMap-v0": BifurcationDefault("init_r", 2.8, 4.0, lambda s: s.x, "x", "samples", 1_000, 200, 5_000),
    "TentMap-v0": BifurcationDefault("init_mu", 1.0, 2.0, lambda s: s.x, "x", "samples", 1_000, 200, 5_000),
    "HenonMap-v0": BifurcationDefault("init_a", 1.0, 1.42, lambda s: s.x, "x", "samples", 1_000, 200, 5_000),
    "IkedaMap-v0": BifurcationDefault("u_param", 0.3, 0.9, lambda s: s.x, "x", "samples", 1_000, 200, 5_000),
    "Rossler-v0": BifurcationDefault("c", 2.0, 6.5, lambda s: s.x[0], "maxima of x", "peaks", 5_000, 6_000, 50_000),
    "Lorenz63-v0": BifurcationDefault("rho", 20.0, 220.0, lambda s: s.x[2], "maxima of z", "peaks", 5_000, 4_000,
                                      20_000),
    "Chua-v0": BifurcationDefault("alpha", 8.0, 16.0, lambda s: s.x[0], "maxima of x", "peaks", 5_000, 6_000, 50_000),
}


def validate() -> None:
    allowed = {"role": ROLES,
               "time_structure": TIME_STRUCTURES,
               "energy": ENERGY,
               "forcing": FORCING,
               "hyperbolicity": HYPERBOLICITY
               }

    for env_id, metadata in ENV_METADATA.items():
        for field, values in allowed.items():
            if getattr(metadata, field) not in values:
                raise ValueError(f"{env_id}: {field} {getattr(metadata, field)!r} is not one of {values}")
