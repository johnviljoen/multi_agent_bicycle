import jax
import numpy as np
import geometry

#### base parameters from which everything else is derived ####
car_params = {
    "wheel_base": 2.8,
    "width": 1.942,
    "front_hang": 0.96,
    "rear_hang": 0.929,
    "max_steer": 0.5,
}

lidar_params = {
    "half_num_beams": 100, # each beam is calculated bidirectionally
    "max_dist": 10
}

# Testing equivalent setup to other hybrid a*

# LB = 2.3
# LF = 2.3
# max_steer = np.deg2rad(40)
# total_length = LB + LF
# wheel_base = 2.7
# width = 1.85
# front_hang = LF - wheel_base/2
# rear_hang = LB - wheel_base/2

# car_params = {
#     "wheel_base": wheel_base,
#     "width": width,
#     "front_hang": front_hang,
#     "rear_hang": rear_hang,
#     "max_steer": max_steer,
# }

#### Derive useful parameters from the base parameters above and add to dict ####

# bubble for fast detection of potential collisions later on
car_params["total_length"] = car_params["rear_hang"] + car_params["wheel_base"] + car_params["front_hang"]
car_params["bubble_radius"] = np.hypot(car_params["total_length"] / 2, car_params["width"] / 2)

# origin is defined around the rear axle, default orientiation is facing east
car_params["corners"] = np.array([
    [car_params["wheel_base"] + car_params["front_hang"], car_params["width"] / 2], # front left
    [- car_params["rear_hang"], car_params["width"] / 2], # back left
    [- car_params["rear_hang"], - car_params["width"] / 2], # back right
    [car_params["wheel_base"] + car_params["front_hang"], - car_params["width"] / 2] # front right
])

car_params["center_to_front"] = car_params["wheel_base"]/2 + car_params["front_hang"]
car_params["center_to_back"] = car_params["wheel_base"]/2 + car_params["rear_hang"]

# calculate the distance from the lidar source to the car exterior to subtract from readings later
lidar_params["angles"] = np.linspace(0, np.pi, lidar_params["half_num_beams"])

default_state = np.array([0,0,0,0])
vertices: jax.numpy.ndarray = geometry.get_corners(default_state, car_params)
halfspaces: jax.numpy.ndarray = geometry.get_halfspace_representation(vertices)

# the values to subtract from the lidar readings later for distances to edge of car
car_params["origin_to_edge"], intersections = geometry.get_dist_to_polygons(default_state, lidar_params["angles"], [vertices], [halfspaces], max_dist=lidar_params["max_dist"])

print('fin')
# car_params["lidar_calibration_dist"] = 

if __name__ == "__main__":

    # test observation
    import lidar
    import scenario
    import jax.numpy as jnp
    
    case_params = scenario.read("data/cases/test_2_agent_case.csv")
    x = jnp.array(case_params["start_poses"])
    lidar.observation(x, case_params, car_params, lidar_params)

    import matplotlib.pyplot as plt

    plt.plot(vertices[:,0], vertices[:,1])
    plt.scatter(intersections[:,0], intersections[:,1])
    plt.savefig('test.png',dpi=500)

    print('fin')