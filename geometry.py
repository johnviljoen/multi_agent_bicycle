import jax.numpy as jnp

def get_halfspace_representation(vertices):
    # repeat the first point so we get all edges between vertices
    vertices = jnp.vstack([vertices, vertices[0]])
    # ax + by + c = 0
    a = vertices[:-1, 1:2] - vertices[1:, 1:2]
    b = vertices[1:, 0:1] - vertices[:-1, 0:1]
    c = vertices[:-1, 0:1] * vertices[1:, 1:2] - vertices[1:, 0:1] * vertices[:-1, 1:2]
    return a, b, c

def get_transform_mat(xi):

    return jnp.array([
        [jnp.cos(xi[2]), -jnp.sin(xi[2]), xi[0]],
        [jnp.sin(xi[2]),  jnp.cos(xi[2]), xi[1]],
        [0, 0, 1]
    ])

def get_rotation_mat(xi):

    return jnp.array([
        [jnp.cos(xi[2]), -jnp.sin(xi[2])],
        [jnp.sin(xi[2]),  jnp.cos(xi[2])]
    ])

def get_corners(xi, car_params):

    rot = get_transform_mat(xi)

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

def get_dist_to_polygons(xi, half_angles, polygon_vertices, polygon_halfspaces, max_dist):

    """calculate all lidar beam distance measurements to the nearest polygon and return the 
    distance and the intersection coordinate in cartesian space. If the distance is larger than
    max_dist we return max dist and the corresponding coordinate

    algorithm:
    - use d = ((v - l0) @ n) / (L2 @ n) to calculate beam distance to halfspace
    - clip d between -max_dist and max_dist to get rid of infinities for parallel lines
    - calculate point of intersection p for all beams with and without valid intersections (GPU requirement)
    - use dot product criterion to determine if intersections are within vertices defining polygon edge
        - if between vertices: 0 < vec_p_v2 @ vec_v1_v2 < vec_v1_v2 @ vec_v1_v2
        - where @ represents the dot operation - draw it out to convince yourself!
        - I apologise for using einsums - they make things more concise and efficient!

    everything else is bookeeping/boilerplate/minor details.

    I am not sure what a more efficient way to do this in jax would be, as we are quite restricted
    by the jit requirements on boolean masks to minimize unecessary compute.
    """
    
    l0 = xi[0:2] # {x,y} position starting the line
    l1 = jnp.array([[jnp.cos(angle), jnp.sin(angle)] for angle in half_angles]) # unit direction vectors
    rot = jnp.eye(2) # get_rotation_mat(xi)
    l1 = l1 @ rot.T # rotate the lidar beams with the car yaw

    d_pos = jnp.ones(len(l1)) * max_dist
    d_neg = jnp.ones(len(l1)) * -max_dist
    intersections = jnp.empty([len(l1) * 2, 2])

    for i, l in enumerate(l1):

        _d_pos = max_dist
        _d_neg = -max_dist

        for v, h in zip(polygon_vertices, polygon_halfspaces):
            n = jnp.hstack([h[0], h[1]]).T # normals of the halfspaces

            # distance to intersection: d = ((v - l0) @ n) / (L2 @ n) # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
            denom = l @ n
            num = jnp.einsum('ij,ji->i', v-l0, n)

            # when lines are parallel then we have infinite distance to intersection
            parallel_mask = denom == 0
            _d = jnp.clip(num/denom, min=-max_dist, max=max_dist)

            # for all nonparallel lines we want to see if they lie between vertices on the polygon
            p = l0 + l * _d[:,None] # the point of intersection
            _v = jnp.vstack([v, v[0]]) # stack vertices for convenience of finding edges
            e = (_v[:-1] - _v[1:]) # edges of polygon
            e_dot_e = jnp.einsum('ij,ij->i', e, e) # edge dotted with itself
            c = v - p # collinear vector on halfspace
            c_dot_e = jnp.einsum('ij,ij->i', c, e) # collinear vector dotted with edge
            intersection_mask = jnp.logical_and(c_dot_e < e_dot_e, 0 < c_dot_e) # draw the geometry to convince yourself
            parallel_or_nonintersecting = jnp.logical_or(parallel_mask, ~intersection_mask)
            d = jnp.where(parallel_or_nonintersecting, max_dist, _d)

            # find closest distances positive and negative - this method of using one distance calculation for 
            # positive and negative necessitates the positive and negative masking below (d>0, d<0), which might
            # be slower than just going around the whole way [0,2pi] rather than [0,pi] as this is.
            _d_pos = jnp.minimum(_d_pos, jnp.min(d, where=d > 0, initial=max_dist))
            _d_neg = jnp.maximum(_d_neg, jnp.max(d, where=d < 0, initial=-max_dist))

        # save the intersections found
        intersections = intersections.at[i*2:i*2+2].set(
            jnp.vstack(
                [l0 + l * _d_pos, 
                 l0 + l * _d_neg]
            )
        )
        d_pos = d_pos.at[i].set(_d_pos)
        d_neg = d_neg.at[i].set(_d_neg)

    distances = jnp.hstack([d_pos, -d_neg]) # stack distances and uninvert negative distances
    return distances, intersections

def get_vertices_and_halfspaces_of_all_cars_except_index(
        x, # joint state
        index, # the index of the car being ignored
        case_params,
        car_params
    ):
    
    car_nums = [i for i in range(case_params["num_cars"])]
    car_nums.remove(index)
    vertices = []
    halfspaces = []
    for i in car_nums:
        _obs_v = get_corners(x[i], car_params)
        ai, bi, ci = get_halfspace_representation(_obs_v)
        vertices.append(_obs_v)
        halfspaces.append([ai, bi, ci])

    return vertices, halfspaces, car_nums

if __name__ == "__main__":

    from params import car_params, lidar_params
    import scenario

    case_params = scenario.read("data/cases/test_2_agent_case.csv")

    

    print('fin')