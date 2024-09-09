import csv
import numpy as np
import matplotlib.pyplot as plt

def get_corners(car_params, x, y, yaw):
    points = np.array([
        [-car_params["rear_hang"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], -car_params["width"] / 2, 1],
        [ car_params["front_hang"] + car_params["wheel_base"], car_params["width"] / 2, 1],
        [-car_params["rear_hang"],  car_params["width"] / 2, 1],
        [-car_params["rear_hang"], -car_params["width"] / 2, 1],
    ]) @ np.array([
        [np.cos(yaw), -np.sin(yaw), x],
        [np.sin(yaw), np.cos(yaw), y],
        [0, 0, 1]
    ]).T
    return points[:, 0:2]

def read(file: str):

    case_params = {}

    with open(file, 'r') as f:
        reader = csv.reader(f)
        tmp = list(reader)
        v = [float(i) for i in tmp[0]]
        
        # Read the number of cars
        num_cars = int(v[0])
        case_params["num_cars"] = num_cars
        
        # Initialize lists for start and goal poses
        case_params["start_poses"] = []
        case_params["goal_poses"] = []
        
        # Read start and goal poses for each car
        offset = 1  # Start reading after the number of cars
        for _ in range(num_cars):
            start_pose = v[offset:offset+3]  # (x0, y0, theta0) for car i
            goal_pose = v[offset+3:offset+6]  # (xf, yf, thetaf) for car i
            case_params["start_poses"].append(start_pose)
            case_params["goal_poses"].append(goal_pose)
            offset += 6  # Move to the next car's data

        # Read obstacle information
        case_params["obs_num"] = int(v[offset])
        num_vertexes = np.array(v[offset+1:offset+1 + case_params["obs_num"]], dtype=int)
        vertex_start = offset + 1 + case_params["obs_num"] + (np.cumsum(num_vertexes, dtype=int) - num_vertexes) * 2
        case_params["obs_v"] = [] # vertex information
        for vs, nv in zip(vertex_start, num_vertexes):
            case_params["obs_v"].append(np.array(v[vs:vs + nv * 2]).reshape((nv, 2), order='A'))
        
        case_params["obs_h"] = [] # halfspace information
        for obs_v in case_params["obs_v"]:
            # repeat the first point so we get all edges between vertices
            obs_v = np.vstack([obs_v, obs_v[0]])
            # ax + by + c = 0
            ai = obs_v[:-1, 1:2] - obs_v[1:, 1:2]
            bi = obs_v[1:, 0:1] - obs_v[:-1, 0:1]
            ci = obs_v[:-1, 0:1] * obs_v[1:, 1:2] - obs_v[1:, 0:1] * obs_v[:-1, 1:2]
            case_params["obs_h"].append([ai, bi, ci])

        # Compute bounds
        all_x = [pose[0] for pose in case_params["start_poses"]] + [pose[0] for pose in case_params["goal_poses"]]
        all_y = [pose[1] for pose in case_params["start_poses"]] + [pose[1] for pose in case_params["goal_poses"]]
        case_params["xmin"] = min(all_x) - 8
        case_params["xmax"] = max(all_x) + 8
        case_params["ymin"] = min(all_y) - 8
        case_params["ymax"] = max(all_y) + 8
        
    return case_params

def write_case_csv(file_name, start_poses, goal_poses, obstacles):
    """
    Write a CSV file that represents a scenario with multiple cars, each with a start pose, goal pose, and obstacles.
    Yes a JSON would be a better fit for this if you want to rework the IO.

    :param file_name: Name of the output CSV file.
    :param start_poses: List of tuples or lists, where each tuple is (x0, y0, yaw0) for the start pose of a car.
    :param goal_poses: List of tuples or lists, where each tuple is (xf, yf, yawf) for the goal pose of a car.
    :param obstacles: List of numpy arrays, where each array has shape [n, 2] representing obstacle vertices.
    """
    assert len(start_poses) == len(goal_poses), "Each car must have both a start pose and a goal pose."

    # Number of cars
    num_cars = len(start_poses)
    
    # Number of obstacles
    obs_num = len(obstacles)
    
    # Number of vertices for each obstacle
    num_vertices = [obs.shape[0] for obs in obstacles]
    
    # Flatten the list of obstacle vertices
    flattened_obstacles = []
    for obs in obstacles:
        flattened_obstacles.extend(obs.flatten())
    
    # Combine start and goal poses for all cars
    cars_data = []
    for start_pose, goal_pose in zip(start_poses, goal_poses):
        cars_data.extend(list(start_pose) + list(goal_pose))
    
    # Combine everything into a single list
    csv_data = [num_cars] + cars_data + [obs_num] + num_vertices + flattened_obstacles
    
    # Write to CSV file
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_data)

def plot_case(case_params, car_params, filename=None, show=False, save=True, bare=False):

    if filename is None:
        filename = 1
    plt.xlim(case_params["xmin"], case_params["xmax"])
    plt.ylim(case_params["ymin"], case_params["ymax"])
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.gca().set_axisbelow(True)

    for j in range(0, case_params["obs_num"]):
        plt.fill(case_params["obs_v"][j][:, 0], case_params["obs_v"][j][:, 1], facecolor = 'red', alpha = 0.5)

    if bare is False:

        for start_pose, goal_pose in zip(case_params["start_poses"], case_params["goal_poses"]):
            x0, y0, yaw0 = start_pose
            xf, yf, yawf = goal_pose
            plt.arrow(x0, y0, np.cos(yaw0), np.sin(yaw0), width=0.2, color = "gold")
            plt.arrow(xf, yf, np.cos(yawf), np.sin(yawf), width=0.2, color = "gold")
            temp = get_corners(car_params, x0, y0, yaw0)
            plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
            temp = get_corners(car_params, xf, yf, yawf)
            plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'red')

        plt.grid(linewidth = 0.2)
        plt.title(filename)
        plt.xlabel('X / m', fontsize = 14)
        plt.ylabel('Y / m', fontsize = 14)
    else:
        plt.axis('off')

    if save is True:
        plt.savefig(f"{filename}.png", dpi=500)
    if show is True:
        plt.show()

if __name__ == "__main__":

    import jax.numpy as jnp
    import jax.random as jr
    from params import car_params

    #### test 2 agent case ####

    obstacles = np.array([
        [ # left obstacle
            [-3.8-2.5, 2.5],
            [-3.8+2.5, 2.5],
            [-3.8+2.5, -2.5],
            [-3.8-2.5, -2.5],
        ],
        [ # right obstacle
            [3.8-2.5, 2.5],
            [3.8+2.5, 2.5],
            [3.8+2.5, -2.5],
            [3.8-2.5, -2.5]
        ],
        [ # top obstacle
            [-6, 10-0.4],
            [6, 10-0.4],
            [6, 10+3.4],
            [-6, 10+3.4]
        ]
    ])

    # Multiple cars, each with start and goal poses
    start_poses = np.array([
        [-7.0, 4.35, np.deg2rad(-10)],
        [6.0, 5.0, np.deg2rad(170)]
    ])
    
    goal_poses = np.array([
        [0.0, 0.0, np.deg2rad(90)],
        [-5.0, 7.5, np.deg2rad(10)]
    ])

    # Write the scenario to a CSV file
    write_case_csv('data/cases/test_2_agent_case.csv', start_poses, goal_poses, obstacles)
    case_params = read("data/cases/test_2_agent_case.csv")
    plot_case(case_params, car_params, filename='data/images/test_2_agent_case', show=False, save=True, bare=False)
    plt.close()

    #### test 4 agent case ####

    # Define obstacles
    obstacles = np.array([
        [  # Top-left parking lot (rectangular area)
            [-12.0, 5.0],
            [-12.0, 1.0],
            [-9.0, 1.0],
            [-9.0, 5.0]
        ],
        [  # Top-right parking lot (rectangular area)
            [9.0, 5.0],
            [9.0, 1.0],
            [12.0, 1.0],
            [12.0, 5.0]
        ],
        [  # Bottom-left parking lot (rectangular area)
            [-12.0, -5.0],
            [-12.0, -1.0],
            [-9.0, -1.0],
            [-9.0, -5.0]
        ],
        [  # Bottom-right parking lot (rectangular area)
            [7.0, -5.0],
            [7.0, -1.0],
            [12.0, -1.0],
            [12.0, -5.0]
        ]
    ])

    # Define start poses of the vehicles
    start_poses = np.array([
        [7.0, -3.0, np.deg2rad(180)],  # Vehicle 0 starting from bottom-right, facing left
        [-7.0, -4.0, np.deg2rad(0)],   # Vehicle 1 starting from bottom-left, facing right
        [7.0, 3.0, np.deg2rad(90)],    # Vehicle 2 starting from top-right, facing upward
        [-7.0, 4.0, np.deg2rad(-90)]   # Vehicle 3 starting from top-left, facing downward
    ])

    # Define goal poses of the vehicles
    goal_poses = np.array([
        [1.0, 1.0, np.deg2rad(90)],    # Vehicle 0 moves to the center, facing upward
        [0.0, -4.0, np.deg2rad(-90)],  # Vehicle 1 moves to the center-bottom, facing downward
        [-3.0, 3.0, np.deg2rad(180)],  # Vehicle 2 moves to center-left, facing left
        [4.0, 2.0, np.deg2rad(0)]      # Vehicle 3 moves to the center-right, facing right
    ])

    # Write the scenario to a CSV file
    write_case_csv('data/cases/test_4_agent_case.csv', start_poses, goal_poses, obstacles)
    case_params = read("data/cases/test_4_agent_case.csv")
    plot_case(case_params, car_params, filename='data/images/test_4_agent_case', show=False, save=True, bare=False)
    plt.close()

    #### test many obstacles ~100 case ####

    obstacle_multiplier = 100
    obstacles = jnp.concat([obstacles]*obstacle_multiplier, axis=0)
    offsets = jr.uniform(jr.PRNGKey(0), minval=-15, maxval=15, shape=(obstacles.shape[0]))[:,None][:,:,None]
    obstacles = obstacles + offsets

    # Define start poses of the vehicles
    start_poses = np.array([
        [-3.0, -3.0, np.deg2rad(180)],  # Vehicle 0 starting from bottom-right, facing left
        [-8.0, -7.0, np.deg2rad(0)],   # Vehicle 1 starting from bottom-left, facing right
        [7.0, 7.0, np.deg2rad(90)],    # Vehicle 2 starting from top-right, facing upward
        [4.0, 4.0, np.deg2rad(-90)]   # Vehicle 3 starting from top-left, facing downward
    ])

    # Write the scenario to a CSV file
    write_case_csv('data/cases/test_many_obstacles.csv', start_poses, goal_poses, obstacles)
    case_params = read("data/cases/test_many_obstacles.csv")
    plot_case(case_params, car_params, filename='data/images/test_many_obstacles', show=False, save=True, bare=False)

    
    print('fin')