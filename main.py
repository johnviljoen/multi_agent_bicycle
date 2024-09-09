
if __name__ == "__main__":

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
    from params import car_params
    from animator import Animator

    seed = 0
    rng = jr.PRNGKey(seed); _rng, rng = jr.split(rng)
    case_params = scenario.read("data/cases/test_4_agent_case.csv")

    num_agents = case_params["num_cars"]
    num_envs = 10000 # 10 # 10000
    Ts = 0.1; Ti=0.0; Tf=10_000.0 # 5.0 # 10_000.0
    times = jnp.arange(Ti, Tf, Ts)
    num_iter = len(times)

    #             {x,   y,   yaw, v   }
    # x = jnp.array([0.0, 0.0, 0.0, 0.0]) # single agent
    # x = jnp.stack([x]*num_agents)       # num agents in one env
    # x = jnp.stack([x]*num_envs)         # num environments of num agents
    # x += jr.normal(_rng, shape=x.shape); _rng, rng = jr.split(rng) # randomize
    x = jnp.array(case_params["start_poses"])
    x = jnp.hstack([x, jnp.zeros([num_agents, 1])]) # add zero velocity
    x = jnp.stack([x]*num_envs)

    #             {acc, steer}
    u = jnp.array([1.0,   1.0])     # single agent
    u = jnp.stack([u]*num_agents)   # num agents in one env
    u = jnp.stack([u]*num_envs)     # num environments of num agents
    u += jr.normal(_rng, shape=u.shape); _rng, rng = jr.split(rng) # randomize
    u_control = jnp.copy(u)

    xdot = functools.partial(dynamics.xdot, car_params=car_params)
    actor = models.StochasticActor([4,32,32,2], _rng); _rng, rng = jr.split(rng)
    critic = models.DoubleCritic([2,32,32,1], _rng); _rng, rng = jr.split(rng)

    xdot_jit = jax.jit(jax.vmap(jax.vmap(xdot)))
    actor_jit = eqx.filter_jit(eqx.filter_vmap(eqx.filter_vmap(actor)))
    critic_jit = eqx.filter_jit(eqx.filter_vmap(eqx.filter_vmap(critic)))
    collision_jit = jax.vmap(functools.partial(collision.rectangle_mask, case_params=case_params, car_params=car_params))

    u = actor_jit(x, jr.split(_rng, num=[num_envs, num_agents])); _rng, rng = jr.split(rng)
    u = actor_jit(x, jr.split(_rng, num=[num_envs, num_agents])); _rng, rng = jr.split(rng)
    x += xdot_jit(x, u) * Ts
    x += xdot_jit(x, u) * Ts
    collision_mask = collision_jit(x)
    collision_mask = collision_jit(x)

    # traj = []
    # not_collisions = []
    # for _ in range(num_iter):
    #     collision_mask = collision_jit(x)
    #     not_collisions.append(jnp.copy(collision_mask[-1]))
    #     u = actor_jit(x, jr.split(_rng, num=[num_envs, num_agents])); _rng, rng = jr.split(rng)
    #     x += xdot_jit(x, u_control) * Ts * collision_mask[:, :, None] # if collided freeze
    #     traj.append(jnp.copy(x[-1]))
    #     c = critic_jit(u)
    # traj = jnp.stack(traj)
    # collisions = ~jnp.vstack(not_collisions)
    # animator = Animator(car_params, case_params, traj, times, collisions)
    # animator.animate()

    def scan_body(carry, _):
        x, _rng, rng = carry
        collision_mask = collision_jit(x)
        u = actor_jit(x, jr.split(_rng, num=[num_envs, num_agents])); _rng, rng = jr.split(rng)
        x += xdot_jit(x, u_control) * Ts * collision_mask[:, :, None] # if collided freeze
        c = critic_jit(u)
        return (x, _rng, rng), None  

    tic = time()
    (x, _, _), _ = jax.lax.scan(scan_body, (x, _rng, rng), None, length=num_iter)
    toc = time()
    scan_time = toc-tic
    print(toc-tic)

    print('fin')


