"""
Description
-----------

To detect collisions in the most efficient way possible
"""

import jax.numpy as jnp
import numpy as np

def _get_rotation_mat(xi):

    return jnp.array([
        [jnp.cos(xi[2]), -jnp.sin(xi[2]), xi[0]],
        [jnp.sin(xi[2]),  jnp.cos(xi[2]), xi[1]],
        [0, 0, 1]
    ])

def _get_corners(xi, car_params):

    rot = _get_rotation_mat(xi)

    # untransformed points
    untransformed_corners = jnp.array([
        [-car_params["rear_hang"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], car_params["width"] / 2, 1],
        [-car_params["rear_hang"],  car_params["width"] / 2, 1]
    ])

    # rotate and translate!
    return (untransformed_corners @ rot)[:,:2]

def _remove_value_from_array(array, value_to_remove):
    mask = array != value_to_remove
    return array[mask]

def _bubble_obstacles(x, case_params, car_params):
    """check collision between car bubbles with either each other or static obstacles
    
    Args:
        x: joint state vector of multi agent env (4x{x,y,yaw,v})
        case_params 
        car_params
    """
    pass

def _rectangle_obstacles(x, case_params, car_params):
    """check collision between rectangle and all other obstacles (including other car rectangles).
    If we are on CPU there are alternative algorithms that are faster (https://en.wikipedia.org/wiki/Point_in_polygon).
    However in a JAX jitted GPU implementation control flow is not allowed if it dictates dynamic shapes - therefore
    I have gone with a projection based approach which can be parallelized on GPU.

    notes: this is non-differentiable unlike OBCA, this is FAST once jitted and vmapped

    Args:
        x: joint state vector of multi agent env (4x{x,y,yaw,v})
        case_params 
        car_params
    """

    # to store if any of the agents is in a state of collision
    collision = 0.0
    car_nums = [i for i in range(case_params["num_cars"])]

    for i, xi in enumerate(x): # iterate through each agent in the env
        
        # corners of current ego car
        corners = _get_corners(xi, car_params)

        # add the other agents to the list of obstacles at this time
        obs_h = case_params["obs_h"].copy()
        obs_v = case_params["obs_v"].copy()
        other_car_nums = car_nums.copy()
        other_car_nums.remove(i)

        for j in other_car_nums:
            _obs_v = _get_corners(x[j], car_params)
            obs_v.append(_obs_v)
            # repeat the first point so we get all edges between vertices
            _obs_v = jnp.vstack([_obs_v, _obs_v[0]])
            # ax + by + c = 0
            ai = _obs_v[:-1, 1:2] - _obs_v[1:, 1:2]
            bi = _obs_v[1:, 0:1] - _obs_v[:-1, 0:1]
            ci = _obs_v[:-1, 0:1] * _obs_v[1:, 1:2] - _obs_v[1:, 0:1] * _obs_v[:-1, 1:2]
            obs_h.append([ai, bi, ci])

        # go through every obstacle for every corner
        for _obs_h in obs_h:
            
            # retrieve halfspace info of polygon: aibi @ x + ci = 0; equiv. to. ax + by + c = 0
            ai = _obs_h[0]
            bi = _obs_h[1]
            ci = _obs_h[2]

            x0 = corners[:,0]
            y0 = corners[:,1]

            sign = ai * x0 + bi * y0 + ci
            inside = (~ jnp.any((sign > 0), axis=0)).sum() # point inside obstacle

            collision += inside

        # check every corner of every obstacle if its inside the agent
        obs_v_vec = jnp.vstack(obs_v)
        _obs_v = jnp.vstack([corners, corners[0]])

        # ax + by + c = 0
        ai = _obs_v[:-1, 1:2] - _obs_v[1:, 1:2]
        bi = _obs_v[1:, 0:1] - _obs_v[:-1, 0:1]
        ci = _obs_v[:-1, 0:1] * _obs_v[1:, 1:2] - _obs_v[1:, 0:1] * _obs_v[:-1, 1:2]

        x0 = obs_v_vec[:,0]
        y0 = obs_v_vec[:,1]

        sign = ai * x0 + bi * y0 + ci
        inside = (~ jnp.any((sign > 0), axis=0)).sum() # point inside obstacle

        collision += inside

    return collision

def collision(x, case_params, car_params):
    pass

if __name__ == "__main__":

    from params import car_params
    from scenario_utils import read
    import jax.random as jr
    import jax
    import functools
    from time import time
    case_params = read("data/cases/test_case.csv")
    num_envs = 100000
    x = jnp.array([
        [0, 0, jnp.deg2rad(20)],
        [2, 5, jnp.deg2rad(-35)]
    ])
    x = jnp.stack([x]*num_envs)
    x = jr.normal(jr.PRNGKey(0), shape=x.shape)

    f = jax.jit(jax.vmap(functools.partial(_rectangle_obstacles, case_params=case_params, car_params=car_params)))
    # f = _rectangle_obstacles
    violation = f(x)
    violation = f(x)

    tic = time()
    violation = f(x)
    toc = time()
    print(toc-tic)

    x = jr.normal(jr.PRNGKey(1), shape=x.shape)
    tic = time()
    violation = f(x)
    toc = time()
    print(toc-tic)

    # now lets plot an environment animation and see this in action

    print('sin')