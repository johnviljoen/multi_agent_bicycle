import jax
import jax.numpy as jnp
import jax.random as jr
import functools

def xdot(x, u, car_params):
    u1_clip = jnp.clip(u[1], a_min=-car_params['max_steer'], a_max=car_params['max_steer'])

    return jnp.hstack([
        x[3]*jnp.cos(x[2]),
        x[3]*jnp.sin(x[2]),
        x[3]*jnp.tan(u1_clip) / car_params['wheel_base'],
        u[0],
    ])

if __name__ == "__main__":

    # how fast can we make this...
    from jax import jit, vmap
    from jax.lax import scan
    from params import car_params
    from tqdm import tqdm
    from time import time
    
    seed = 0
    rng = jr.PRNGKey(seed); _rng, rng = jr.split(rng)
    num_agents = 4
    num_envs = 10_000
    Ts = 0.1
    n_iter = 100_000

    #             {x,   y,   yaw, v   }
    x = jnp.array([0.0, 0.0, 0.0, 0.0]) # single agent
    x = jnp.stack([x]*num_agents)       # num agents in one env
    x = jnp.stack([x]*num_envs)         # num environments of num agents
    x += jr.normal(_rng, shape=x.shape); _rng, rng = jr.split(rng) # randomize

    #             {acc, steer}
    u = jnp.array([1.0,   1.0])     # single agent
    u = jnp.stack([u]*num_agents)   # num agents in one env
    u = jnp.stack([u]*num_envs)     # num environments of num agents
    u += jr.normal(_rng, shape=u.shape); _rng, rng = jr.split(rng) # randomize
    
    # supply car_params to this function
    f = functools.partial(xdot, car_params=car_params)
    vec_f = jit(vmap(vmap(f)))

    x += vec_f(x, u) * Ts

    def scan_body(carry, _):
        x = carry
        x_new = x + vec_f(x, u) * Ts
        return x_new, None        

    tic = time()
    x, _ = scan(scan_body, x, None, length=n_iter)
    toc = time()
    scan_time = toc-tic
    print(toc-tic)

    tic = time()
    for i in tqdm(range(n_iter)):
        x += vec_f(x, u) * Ts
    toc = time()
    loop_time = toc-tic
    print(toc-tic)

    print(f"fps per vehicle for scan: {n_iter * num_envs / scan_time}")
    print(f"fps per vehicle for loop: {n_iter * num_envs / loop_time}")

    # seem to get ~1.4 billion FPS fopr scan, ~91 million FPS for loop



