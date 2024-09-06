"""
Description
-----------

To detect collisions in the most efficient way possible
"""

import jax.numpy as jnp

def _bubble_obstacles(x, case_params, car_params):
    """check collision between car bubbles with either each other or static obstacles
    
    Args:
        x: joint state vector of multi agent env (4x{x,y,yaw,v})
        case_params 
        car_params
    """
    pass

def _rectangle_obstacles(x, case_params, car_params):
    """check collision between rectangle and all other obstacles (including other car rectangles)

    Args:
        x: joint state vector of multi agent env (4x{x,y,yaw,v})
        case_params 
        car_params
    """
    for xi in x: # iterate through each agent in the env

        # rotation matrix for the vehicle
        rot = jnp.array([
            [jnp.cos(xi[2]), -jnp.sin(xi[2])],
            [jnp.sin(xi[2]),  jnp.cos(xi[2])]
        ])

def collision(x, case_params, car_params):
    pass

if __name__ == "__main__":


    print('sin')