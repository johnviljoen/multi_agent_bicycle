import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scenario import get_corners

class Animator:
    def __init__(
            self,
            car_params,
            case_params,
            traj,
            times,
            collisions = None,
            max_frames = 500,
            Ts = 0.1,
            save_path = 'data/gifs', # None
            save_name = None,     
            fps = 10,
            dpi = 125,              
        ):

        self.traj = traj
        self.Ts = Ts
        self.fps = fps
        self.dpi = dpi
        self.car_params = car_params
        self.case_params = case_params


        num_steps = len(times)
        def compute_render_interval(num_steps, max_frames):
            render_interval = 1  # Start with rendering every frame.
            # While the number of frames using the current render interval exceeds max_frames, double the render interval.
            while num_steps / render_interval > max_frames:
                render_interval *= 2
            return render_interval
        render_interval = compute_render_interval(num_steps, max_frames)

        self.traj = traj[::render_interval,:]
        self.times = times[::render_interval]
        if collisions is not None:
            self.collisions = collisions[::render_interval]
        else:
            self.collisions = np.zeros_like(self.times) # no collisions
        self.save_path = save_path

        if save_name is None:
            self.save_name = "test" # datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.save_name = save_name

        # Instantiate the figure
        # ----------------------
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot() # projection='2d')
        # placeholder for timestamp plotting
        self.title_time = self.ax.text(0.05, 0.95, "", transform=self.ax.transAxes)

        # plot all static elements
        # ------------------------
        self.ax.set_xlim(case_params["xmin"], case_params["xmax"])
        self.ax.set_ylim(case_params["ymin"], case_params["ymax"])
        self.ax.set_aspect('equal', adjustable = 'box')
        self.ax.set_axisbelow(True)
        self.ax.set_title('Case %d') # % (i + 1))
        self.ax.grid(linewidth = 0.2)
        self.ax.set_xlabel('X / m', fontsize = 14)
        self.ax.set_ylabel('Y / m', fontsize = 14)

        for j in range(0, case_params["obs_num"]):
            self.ax.fill(case_params["obs_v"][j][:, 0], case_params["obs_v"][j][:, 1], facecolor = 'k', alpha = 0.5)

        for pose in case_params["start_poses"]:
            self.ax.arrow(pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]), width=0.2, color = "gold")
            temp = get_corners(car_params, pose[0], pose[1], pose[2])
            self.ax.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
        for pose in case_params["goal_poses"]:
            self.ax.arrow(pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]), width=0.2, color = "gold")
            temp = get_corners(car_params, pose[0], pose[1], pose[2])
            self.ax.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'red')

        # create the lines used for dynamic elements
        # ------------------------------------------
        self.lines = []
        for _ in range(case_params["num_cars"]):
            self.lines.append(self.ax.plot([], [], lw=2, color='black')[0])

    def update_lines(self, i):

        joint_state = self.traj[i]
        collision = self.collisions[i]
        for i, state in enumerate(joint_state):
            corners = get_corners(self.car_params, state[0], state[1], state[2])
            self.lines[i].set_data(corners[:,0], corners[:,1])
            if collision[i] == True:
                self.lines[i].set_color('red')
            else:
                self.lines[i].set_color('green')

        self.title_time.set_text(u"Time = {:.2f} s".format(self.times[i]))

        return self.lines
    
    def ini_plot(self):
        for i in range(self.case_params["num_cars"]):
            self.lines[i].set_data(np.empty([1]), np.empty([1]))
        return self.lines

    def animate(self):
        line_ani = animation.FuncAnimation(
            self.fig,
            self.update_lines,
            init_func=self.ini_plot,
            frames=len(self.times)-1,
            interval=self.Ts*10,
            blit=False
        )

        if self.save_path is not None:
            line_ani.save(f'{self.save_path}/{self.save_name}.gif', dpi=self.dpi, fps=self.fps)
            # Update the figure with the last frame of animation
            self.update_lines(len(self.times[1:])-1)
            # Save the final frame as an SVG for good paper plots
            self.fig.savefig(f'{self.save_path}/{self.save_name}_final_frame.svg', format='svg')
 
        # plt.show()
        return line_ani

if __name__ == "__main__":

    from params import car_params
    from scenario import read

    case_params = read("data/cases/test_case.csv")

    traj = np.random.normal(size=(10,2,4)) # 10 timesteps, 2 agents, 4 states
    times = np.arange(0,1,0.1)

    animator = Animator(car_params, case_params, traj, times)
    animator.animate()