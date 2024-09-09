import jax.numpy as jnp

def get_rotation_mat(xi):

    return jnp.array([
        [jnp.cos(xi[2]), -jnp.sin(xi[2]), xi[0]],
        [jnp.sin(xi[2]),  jnp.cos(xi[2]), xi[1]],
        [0, 0, 1]
    ])

def get_corners(xi, car_params):

    rot = get_rotation_mat(xi)

    # untransformed points
    untransformed_corners = jnp.array([
        [-car_params["rear_hang"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], car_params["width"] / 2, 1],
        [-car_params["rear_hang"],  car_params["width"] / 2, 1]
    ])

    # rotate and translate!
    return (untransformed_corners @ rot.T)[:,:2]

def overlap(points, halfspaces):
    # retrieve halfspace info of polygon: aibi @ x + ci = 0; equiv. to. ax + by + c = 0
    ai = halfspaces[0]
    bi = halfspaces[1]
    ci = halfspaces[2]

    x0 = points[:,0]
    y0 = points[:,1]

    sign = ai * x0 + bi * y0 + ci

    in_halfspaces = jnp.logical_or(jnp.all((sign > 0), axis=0), jnp.all((sign < 0), axis=0))
    any_in_halfspaces = jnp.any(in_halfspaces)

    return any_in_halfspaces

def get_line_to_polygon_dist(line, polygon):

    raise NotImplementedError