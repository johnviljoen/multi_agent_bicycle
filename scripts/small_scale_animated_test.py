

import functools
from time import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import scenario
import collision
import models
import dynamics
import lidar
from params import car_params, lidar_params
from animator import Animator

seed = 0
rng = jr.PRNGKey(seed); _rng, rng = jr.split(rng)
# case_params = scenario.read("data/cases/test_many_obstacles.csv")
case_params = scenario.read("data/cases/test_4_agent_case.csv")
# case_params = scenario.read("data/cases/test_2_agent_case.csv")
num_envs=10; Ti=0.0; Tf=5.0; Ts=0.1
xdot = functools.partial(dynamics.xdot, car_params=car_params)
# obs_size = lidar_params["half_num_beams"] * 2 + 4
obs_size = 4
actor = models.StochasticActor([obs_size,32,32,2], _rng); _rng, rng = jr.split(rng)
critic = models.DoubleCritic([2,32,32,1], _rng); _rng, rng = jr.split(rng)

# run_small_scale()
rng = jr.PRNGKey(seed); _rng, rng = jr.split(rng)
num_agents = case_params["num_cars"]

x = jnp.array(case_params["start_poses"])
x = jnp.hstack([x, jnp.zeros([num_agents, 1])]) # add zero velocity
x = jnp.stack([x]*num_envs)

#             {acc, steer}
u = jnp.array([1.0,   1.0])     # single agent
u = jnp.stack([u]*num_agents)   # num agents in one env
u = jnp.stack([u]*num_envs)     # num environments of num agents
u += jr.normal(_rng, shape=u.shape); _rng, rng = jr.split(rng) # randomize
u_control = jnp.copy(u)

times = jnp.arange(Ti, Tf, Ts)
num_iter = len(times)
xdot_jit = jax.jit(jax.vmap(jax.vmap(xdot)))
actor_jit = eqx.filter_jit(eqx.filter_vmap(eqx.filter_vmap(actor)))
critic_jit = eqx.filter_jit(eqx.filter_vmap(eqx.filter_vmap(critic)))
collision_jit = jax.jit(jax.vmap(functools.partial(collision.rectangle_mask, case_params=case_params, car_params=car_params)))
# observation_jit = jax.jit(jax.vmap(functools.partial(lidar.distance_observation, case_params=case_params, car_params=car_params, lidar_params=lidar_params)))

tic = time()

# d = observation_jit(x)
# d = observation_jit(x)
# y = jnp.concat([x,d], axis=2)
y = x
u = actor_jit(y, jr.split(_rng, num=[num_envs, num_agents])); _rng, rng = jr.split(rng)
u = actor_jit(y, jr.split(_rng, num=[num_envs, num_agents])); _rng, rng = jr.split(rng)
x += xdot_jit(x, u) * Ts
x += xdot_jit(x, u) * Ts
collision_mask = collision_jit(x)
collision_mask = collision_jit(x)

toc = time()
print(f"compilation time: {toc - tic}")

traj = []
not_collisions = []
for _ in range(num_iter):
    collision_mask = collision_jit(x)
    not_collisions.append(jnp.copy(collision_mask[-1]))
    # y = jnp.concat([x,d], axis=2)
    y = x
    u = actor_jit(y, jr.split(_rng, num=[num_envs, num_agents])); _rng, rng = jr.split(rng)
    x += xdot_jit(x, u_control) * Ts * collision_mask[:, :, None] # if collided freeze
    # d = observation_jit(x)
    # intersections.append(beam_hit_coords)
    traj.append(jnp.copy(x[-1]))
    c = critic_jit(u)
traj = jnp.stack(traj)
collisions = ~jnp.vstack(not_collisions)
print('instantiating animator...')
animator = Animator(car_params, case_params, lidar_params, traj, times, collisions)
print('animating...')
animator.animate()
print('fin')


