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
    return (untransformed_corners @ rot.T)[:,:2]

def _overlap(points, halfspaces):
    # retrieve halfspace info of polygon: aibi @ x + ci = 0; equiv. to. ax + by + c = 0
    ai = halfspaces[0]
    bi = halfspaces[1]
    ci = halfspaces[2]

    x0 = points[:,0]
    y0 = points[:,1]

    sign = ai * x0 + bi * y0 + ci

    in_halfspaces = jnp.all((sign > 0), axis=0)
    any_in_halfspaces = jnp.any(in_halfspaces)

    return any_in_halfspaces # (~ jnp.any((sign > 0), axis=0)).sum() # point inside obstacle

def _rectangle_obstacles(x, case_params, car_params):
    """check collision between rectangle and all other obstacles (including other car rectangles).
    If we are on CPU there are alternative algorithms that are faster (https://en.wikipedia.org/wiki/Point_in_polygon).
    However in a JAX jitted GPU implementation control flow is not allowed if it dictates dynamic shapes - therefore
    I have gone with a projection based approach which can be parallelized on GPU.

    we define a collision matrix data structure as follows:

    mat = |o i i i| 
          |i o i i|
          |i i o i|
          |i i i o|

    where diagonal elements at {i,i} represent if car {i} is colliding with any obstacle,
    and off diagonal elements {i,j} represent if a car {i} is colliding with another car {j}.
    Note that {i,j} collision does not necessarily mean {j,i} collision as collision in this
    case is detected by a corner of {i} being within {j}. It is sufficient to test {i,j} and
    {j,i} for collision in this way to detect if a collision has occured.

    Args:
        x: joint state vector of multi agent env (4x{x,y,yaw,v})
        case_params 
        car_params
    Returns: 
        collision_matrix: as described above
    """

    collision_matrix = jnp.empty([case_params["num_cars"], case_params["num_cars"]])
    car_nums = [i for i in range(case_params["num_cars"])]

    for i, xi in enumerate(x): # iterate through each agent in the env
        
        # corners of current ego car
        corners = _get_corners(xi, car_params)

        # add the other agents to the list of obstacles at this time
        other_car_nums = car_nums.copy()
        other_car_nums.remove(i)
        car_v = []
        car_h = []
        for j in other_car_nums:
            _obs_v = _get_corners(x[j], car_params)
            car_v.append(_obs_v)
            # repeat the first point so we get all edges between vertices
            _obs_v = jnp.vstack([_obs_v, _obs_v[0]])
            # ax + by + c = 0
            ai = _obs_v[:-1, 1:2] - _obs_v[1:, 1:2]
            bi = _obs_v[1:, 0:1] - _obs_v[:-1, 0:1]
            ci = _obs_v[:-1, 0:1] * _obs_v[1:, 1:2] - _obs_v[1:, 0:1] * _obs_v[:-1, 1:2]
            car_h.append([ai, bi, ci])

        # check collision with other cars - only need to do one way test as we loop over all cars
        for j, k in zip(other_car_nums, range(len(other_car_nums))):
            inside = _overlap(corners, car_h[k])
            collision_matrix = collision_matrix.at[i,j].set(collision_matrix[i,j] + inside)

        # go through every obstacle for every corner
        for _obs_h in case_params["obs_h"]:
            inside = _overlap(corners, _obs_h)
            collision_matrix = collision_matrix.at[i,i].set(collision_matrix[i,i] + inside)

        # check every corner of every obstacle if its inside the agent
        obs_v_vec = jnp.vstack(case_params["obs_v"])

        # get the halfspaces for the current ego vehicle
        _obs_v = jnp.vstack([corners, corners[0]])
        ai = _obs_v[:-1, 1:2] - _obs_v[1:, 1:2]
        bi = _obs_v[1:, 0:1] - _obs_v[:-1, 0:1]
        ci = _obs_v[:-1, 0:1] * _obs_v[1:, 1:2] - _obs_v[1:, 0:1] * _obs_v[:-1, 1:2]

        inside = _overlap(obs_v_vec, [ai, bi, ci])
        collision_matrix = collision_matrix.at[i,i].set(collision_matrix[i,i] + inside)

    return collision_matrix

def _bubble_obstacles(x, case_params, car_params):
    """check collision between car bubbles with either each other or static obstacles
    
    Args:
        x: joint state vector of multi agent env (4x{x,y,yaw,v})
        case_params 
        car_params
    """
    pass

def rectangle_mask(x, case_params, car_params):

    """use the rectangle obstacle collision algorithm from above to form a mask which we can multiply the 
    state update function by to freeze collided entities. This functionality is separated out from the 
    collision matrix calculation to allow for matrix analysis separately.
    """

    collision_matrix = _rectangle_obstacles(x, case_params, car_params)
    collision_mask_vert = jnp.all(collision_matrix == 0, axis=0)
    collision_mask_horz = jnp.all(collision_matrix == 0, axis=1)
    return jnp.logical_or(collision_mask_vert, collision_mask_horz)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from params import car_params
    from scenario import read
    import jax.random as jr
    import jax
    import functools
    from time import time
    
    case_params = read("data/cases/test_case.csv")
    num_envs = 10
    x = jnp.array([
        [0, 0, jnp.deg2rad(20)],
        [0, 2, jnp.deg2rad(20)]
    ])
    x = jnp.stack([x]*num_envs)
    x = jr.normal(jr.PRNGKey(0), shape=x.shape)

    f = jax.jit(jax.vmap(functools.partial(_rectangle_obstacles, case_params=case_params, car_params=car_params)))
    # f = jax.vmap(functools.partial(_rectangle_obstacles, case_params=case_params, car_params=car_params))
    # f = functools.partial(_rectangle_obstacles, case_params=case_params, car_params=car_params)
    # x = x[0]
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

    # print(violation)
    jx = x[0]
    collision_matrix = _rectangle_obstacles(jx, case_params, car_params)

    for xi in jx:
        corners = _get_corners(xi, car_params)
        plt.plot(corners[:,0], corners[:,1], 'black')
    plt.savefig('test.png', dpi=500)



    # now lets plot an environment animation and see this in action
    # print(violation)
    
    print('fin')