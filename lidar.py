import jax.numpy as jnp
import geometry

def distance_observation(x, case_params, car_params, lidar_params):

    """this is largely a wrapper around geometry.get_dist_to_polygons function
    to ensure compatibility with the scene description I am using - if you are
    interested in how the lidar calculation algorithm works you should look there!
    
    returns the distance measurements of the lidar
    """

    static_vertices = case_params["obs_v"]
    static_halfspaces = case_params["obs_h"]
    distances = jnp.empty([len(x), lidar_params["half_num_beams"]*2])

    for i, xi in enumerate(x):

        dynamic_vertices, dynamic_halfspaces, _ = \
            geometry.get_vertices_and_halfspaces_of_all_cars_except_index(x, i, case_params, car_params)

        vertices = static_vertices + dynamic_vertices
        halfspaces = static_halfspaces + dynamic_halfspaces

        d, _ = geometry.get_dist_to_polygons(
            xi, lidar_params["angles"], vertices, halfspaces, lidar_params["max_dist"]
        )

        distances = distances.at[i].set(d - car_params["origin_to_edge"])

    return distances

def coordinate_observation(x, case_params, car_params, lidar_params):

    """this is largely a wrapper around geometry.get_dist_to_polygons function
    to ensure compatibility with the scene description I am using - if you are
    interested in how the lidar calculation algorithm works you should look there!
    
    returns the coordinates of intersection of the lidar beams with obstacles
    """

    static_vertices = case_params["obs_v"]
    static_halfspaces = case_params["obs_h"]
    intersections = jnp.empty([len(x), lidar_params["half_num_beams"]*2, 2])

    for i, xi in enumerate(x):

        dynamic_vertices, dynamic_halfspaces, _ = \
            geometry.get_vertices_and_halfspaces_of_all_cars_except_index(x, i, case_params, car_params)

        vertices = static_vertices + dynamic_vertices
        halfspaces = static_halfspaces + dynamic_halfspaces

        _, o = geometry.get_dist_to_polygons(
            xi, lidar_params["angles"], vertices, halfspaces, lidar_params["max_dist"]
        )

        intersections = intersections.at[i].set(o)

    return intersections

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import jax.random as jr
    import jax
    import functools
    from animator import Animator
    
    import dynamics
    import collision

    import scenario
    from params import car_params, lidar_params

    case_params = scenario.read("data/cases/test_4_agent_case.csv")

    scenario.plot_case(case_params, car_params, save=False)

    x = jnp.array(case_params["start_poses"])

    intersections = coordinate_observation(x, case_params, car_params, lidar_params)

    named_colors = list(mcolors.CSS4_COLORS.keys())

    for i, ii in enumerate(intersections):

        plt.scatter(ii[:,0], ii[:,1], color=named_colors[i])

    plt.savefig("test.png", dpi=500)

    Ti, Tf, Ts = 0.0, 5.0, 0.1
    num_agents = 4
    num_envs = 2
    rng = jr.PRNGKey(0); _rng, rng = jr.split(rng)

    #             {acc, steer}
    u = jnp.array([1.0,   1.0])     # single agent
    u = jnp.stack([u]*num_agents)   # num agents in one env
    u = jnp.stack([u]*num_envs)     # num environments of num agents
    u += jr.normal(_rng, shape=u.shape); _rng, rng = jr.split(rng) # randomize
    u_control = jnp.copy(u)

    x = jnp.array(case_params["start_poses"])
    x = jnp.hstack([x, jnp.zeros([num_agents, 1])]) # add zero velocity
    x = jnp.stack([x]*num_envs)

    times = jnp.arange(Ti, Tf, Ts)
    num_iter = len(times)
    xdot_jit = jax.jit(jax.vmap(jax.vmap(functools.partial(dynamics.xdot, car_params=car_params))))
    collision_jit = jax.jit(jax.vmap(functools.partial(collision.rectangle_mask, case_params=case_params, car_params=car_params)))
    # observation_jit = jax.jit(jax.vmap(functools.partial(lidar.distance_observation, case_params=case_params, car_params=car_params, lidar_params=lidar_params)))

    traj = []
    not_collisions = []
    for _ in range(num_iter):
        collision_mask = collision_jit(x)
        not_collisions.append(jnp.copy(collision_mask[-1]))
        x += xdot_jit(x, u_control) * Ts * collision_mask[:, :, None] # if collided freeze
        traj.append(jnp.copy(x[-1]))
    traj = jnp.stack(traj)
    collisions = ~jnp.vstack(not_collisions)
    print('instantiating animator...')
    animator = Animator(car_params, case_params, lidar_params, traj, times, collisions)
    print('animating...')
    animator.animate()
    print('fin')

    print('fin')