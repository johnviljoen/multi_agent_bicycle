import jax.numpy as jnp

def get_halfspace_representation(vertices):
    # repeat the first point so we get all edges between vertices
    vertices = jnp.vstack([vertices, vertices[0]])
    # ax + by + c = 0
    a = vertices[:-1, 1:2] - vertices[1:, 1:2]
    b = vertices[1:, 0:1] - vertices[:-1, 0:1]
    c = vertices[:-1, 0:1] * vertices[1:, 1:2] - vertices[1:, 0:1] * vertices[:-1, 1:2]
    return a, b, c

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

def get_dist_to_polygons(xi, half_angles, polygon_vertices, polygon_halfspaces, max_dist):

    """distance to the closest polygon in a list of polygons of a line with a given start position,
    with a max distance given.
    """
    
    l0 = xi[0:2] # {x,y} position starting the line
    l1 = jnp.array([[jnp.cos(angle), jnp.sin(angle)] for angle in half_angles]) # unit direction vectors

    d_pos = jnp.ones(len(l1)) * max_dist
    d_neg = jnp.ones(len(l1)) * -max_dist
    intersections = jnp.empty([len(l1) * 2, 2])

    for i, l in enumerate(l1):

        _d_pos = max_dist
        _d_neg = -max_dist

        for v, h in zip(polygon_vertices, polygon_halfspaces):
            n = jnp.hstack([h[0], h[1]]).T # normals of the halfspaces

            # d = ((v - l0) @ n) / (L2 @ n) # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
            d = jnp.empty(len(v))
            denom = l @ n
            num = jnp.einsum('ij,ji->i', v-l0, n)
            
            # when the lines are parallel the denominator is zero - set d to max distance
            nonparallel_mask = denom != 0
            d = d.at[~nonparallel_mask].set(max_dist)

            # when lines are nonparallel we calculate the distance
            nonparallel_distances = num[nonparallel_mask]/denom[nonparallel_mask]
            clipped_nonparallel_distances = jnp.clip(nonparallel_distances, min=-max_dist, max=max_dist)
            d = d.at[nonparallel_mask].set(clipped_nonparallel_distances)
        
            # we need to see if the intersection actually lies on the polygon - i.e. if there is a convex combination
            # of the vertices defining the halfspace of the polygon that contains the intersection
            p = l0 + l * d[nonparallel_mask][:,None]
            v_cyclic = jnp.vstack([v, v[0]])
            edges = (v_cyclic[:-1] - v_cyclic[1:])[nonparallel_mask]
            edges_dot_edges = jnp.einsum('ij,ij->i', edges, edges)
            collinear_vec = v[nonparallel_mask] - p
            vec_dot_edge = jnp.einsum('ij,ij->i', collinear_vec, edges)
            intersection_mask = jnp.logical_and(vec_dot_edge < edges_dot_edges, 0 < vec_dot_edge)
            nonparallel_nonintersecting_mask = nonparallel_mask
            nonparallel_nonintersecting_mask = nonparallel_nonintersecting_mask.at[nonparallel_mask].set(~intersection_mask)
            d = d.at[nonparallel_nonintersecting_mask].set(max_dist)

            # find closest distances positive and negative - this method of using one distance calculation for 
            # positive and negative necessitates the positive and negative masking below (d>0, d<0), which might
            # be slower than just going around the whole way [0,2pi] rather than [0,pi] as this is.
            _d_pos = min(_d_pos, jnp.min(d[d>0]))
            _d_neg = max(_d_neg, jnp.max(d[d<0]))

        # save the intersections found
        intersections = intersections.at[i*2:i*2+2].set(
            jnp.vstack(
                [l0 + l * _d_pos, 
                 l0 + l * _d_neg]
            )
        )
        
        d_pos = d_pos.at[i].set(_d_pos)
        d_neg = d_neg.at[i].set(_d_neg)

    distances = jnp.hstack([d_pos, -d_neg])

    return distances, intersections


def alt_get_dist_to_polygons(xi, angles, polygon_vertices, polygon_halfspaces, max_dist):

    """
    UNFINISHED - might be faster than the above - might be slower

    distance to the closest polygon in a list of polygons of a line with a given start position,
    with a max distance given.
    """
    
    l0 = xi[0:2] # {x,y} position starting the line
    l1 = jnp.array([[jnp.cos(angle), jnp.sin(angle)] for angle in angles]) # unit direction vectors

    distances = jnp.ones(len(l1)) * max_dist

    for i, l in enumerate(l1):

        _d = max_dist

        for v, h in zip(polygon_vertices, polygon_halfspaces):
            n = jnp.hstack([h[0], h[1]]).T # normals of the halfspaces

            # d = ((v - l0) @ n) / (L2 @ n) # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
            d = jnp.empty(len(v))
            denom = l @ n
            num = jnp.einsum('ij,ji->i', v-l0, n)
            
            # when the lines are parallel the denominator is zero - set d to max distance
            nonparallel_mask = denom != 0

            d = d.at[~nonparallel_mask].set(max_dist)

            # when lines are nonparallel we calculate the distance
            nonparallel_distances = num[nonparallel_mask]/denom[nonparallel_mask]
            clipped_nonparallel_distances = jnp.clip(nonparallel_distances, min=-max_dist, max=max_dist)
            
            # when distance is negative we ignore - as we go whole way around circle
            pos_mask = clipped_nonparallel_distances > 0
            d = d.at[nonparallel_mask].set(clipped_nonparallel_distances)
        
            # find closest distances positive and negative - this method of using one distance calculation for 
            # positive and negative necessitates the positive and negative masking below (d>0, d<0), which might
            # be slower than just going around the whole way [0,2pi] rather than [0,pi] as this is.
            _d_pos = min(_d_pos, jnp.min(d[d>0]))
            _d_neg = max(_d_neg, jnp.max(d[d<0]))
        
        d_pos = d_pos.at[i].set(_d_pos)
        d_neg = d_neg.at[i].set(_d_neg)

    distances = jnp.hstack([d_pos, -d_neg])


if __name__ == "__main__":

    from params import car_params, lidar_params
    import scenario

    case_params = scenario.read("data/cases/test_2_agent_case.csv")

    

    print('fin')