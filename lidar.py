import jax.numpy as jnp
import geometry

def observation(x, case_params, car_params, lidar_params):

    """this is largely a wrapper around geometry.get_dist_to_polygons function
    to ensure compatibility with the scene description I am using - if you are
    interested in how the lidar calculation algorithm works you should look there!
    """

    static_vertices = case_params["obs_v"]
    static_halfspaces = case_params["obs_h"]
    car_nums = [i for i in range(case_params["num_cars"])]

    distances = jnp.empty([len(x), lidar_params["half_num_beams"]*2])

    for i, xi in enumerate(x):

        # add the other agents to the list of obstacles at this time
        other_car_nums = car_nums.copy()
        other_car_nums.remove(i)
        dynamic_vertices = []
        dynamic_halfspaces = []
        for j in other_car_nums:
            _obs_v = geometry.get_corners(x[j], car_params)
            ai, bi, ci = geometry.get_halfspace_representation(_obs_v)
            dynamic_vertices.append(_obs_v)
            dynamic_halfspaces.append([ai, bi, ci])

        vertices = static_vertices + dynamic_vertices
        halfspaces = static_halfspaces + dynamic_halfspaces

        d, _ = geometry.get_dist_to_polygons(
            xi, lidar_params["angles"], vertices, halfspaces, lidar_params["max_dist"]
        )

        distances = distances.at[i].set(d - car_params["origin_to_edge"])

    return distances
