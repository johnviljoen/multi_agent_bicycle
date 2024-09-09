import numpy as np

# base parameters from which everything else is derived
car_params = {
    "wheel_base": 2.8,
    "width": 1.942,
    "front_hang": 0.96,
    "rear_hang": 0.929,
    "max_steer": 0.5,
}

lidar_params = {
    "num_beams": 200,
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
angles = np.linspace(0, 2*np.pi, lidar_params["num_beams"])

# car_params["lidar_calibration_dist"] = 


