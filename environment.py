#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import ipdb
import random
import numpy as np
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Team Members:
William Rich

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Obstacle():

    """
    An obstacle in the configuration space.
    """

    def __init__(self, top_left, bottom_right):

        """
        top_left: (x, y)
        bottom_right: (x, y)
        these are the coordinates of the obstacle
        """

        self.top_left = top_left
        self.bottom_right = bottom_right

        self.occupied_region = [] #list of points that the obstacle occupies
        self.get_occupied_region() #get the area that the obstacle covers in the configuration space

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_occupied_region(self):
        """
        Determine what region in the configuration space is occupied by the obstacle.
        """
        for x_point in range(self.top_left[0], self.bottom_right[0] + 1): #sweep the x range
            for y_point in range(self.top_left[1], self.bottom_right[1] + 1): #sweep the y range
                self.occupied_region.append((x_point, y_point))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class environment:

    """

    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, horz_size, vert_size, obstacles, start_point, goal_point):

        self.horz_size = horz_size
        self.vert_size = vert_size
        self.obstacles = self.build_all_obstacles(obstacles)
        self.occupied_region = self.determine_global_occupied_region()
        self.map = np.zeros((self.vert_size, self.horz_size))
        self.agent_state = start_point[0], start_point[1], 0
        self.start_point = start_point
        self.goal_point = goal_point

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build_all_obstacles(self, obstacles):

        """
        Initialise instances of each of the obstacles using the obstacle class.
        """

        obstacle_objects = {}
        for id, obstacle in enumerate(obstacles):
            name = 'obstacle_' + str(id)
            obstacle_objects[name] = Obstacle(obstacle[0], obstacle[1])

        return obstacle_objects

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def determine_global_occupied_region(self):

        """
        Determine the global region occupied by the obstacles.
        """

        occupied_region = []
        for obstacle_name in self.obstacles.keys():
            for point in self.obstacles[obstacle_name].occupied_region:
                occupied_region.append(point)
        occupied_region = list(set(occupied_region)) #make unique

        return occupied_region

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def plot_scene(self):

        """
        for validation, plot the scene
        """

        state = self.get_state()

        plt.imshow(state)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def act(self, action):

        """
        Move, reward the agent.
        """

        state = self.move_agent(action)
        reward, done = self.get_reward()

        return state, reward, done

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_reward(self):

        """
        Determine if rewarded.
        """

        if self.agent_state[0] == self.goal_point[0] and self.agent_state[1] == self.goal_point[1]:
            reward = 1
            done = True

        else:
            reward = -1
            done = False

        return reward, done

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def move_agent(self, action):

        action = action[0]
        x_s = -1 if action[0] < 0 else 1
        y_s = -1 if action[1] < 0 else 1

        """
        Make the action an integer
        """

        action = (math.ceil(abs(action[0])) * x_s, math.ceil(abs(action[1])) * y_s)

        for h_step in range(self.agent_state[0], self.agent_state[0] + action[0] + x_s, x_s):
            for v_step in range(self.agent_state[1], self.agent_state[1] + action[1] + y_s, y_s):

                if ((h_step, v_step) in self.occupied_region) or ((h_step < 0) or (v_step < 0) or (h_step >= self.horz_size) or (v_step >= self.vert_size)):

                    px = (h_step)
                    py = (v_step - y_s)
                    pz = 0
                    if px < 0:
                        px = 0
                    if py < 0:
                        py = 0
                    if px >= self.horz_size:
                        px = self.horz_size - 1
                    if py >= self.vert_size:
                        py = self.vert_size - 1

                    self.agent_state = px, py, pz
                    return self.get_state()

        self.agent_state = self.agent_state[0] + action[0], self.agent_state[1] + action[1], 0
        return self.get_state()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_state(self):

        """
        Prepare an array of the state.
        """

        #set up the arena
        state = np.ones((self.vert_size, self.horz_size))

        #plot the obstacles
        for point in self.occupied_region:
            state[self.vert_size - point[1] - 1, point[0] - 1] = 0

        for pointx in [self.goal_point[0] -1, self.goal_point[0], self.goal_point[0] + 1]:
            for pointy in [self.goal_point[1] -1, self.goal_point[1], self.goal_point[1] + 1]:
                if pointx not in [-1, self.horz_size] and pointy not in [-1, self.vert_size]:
                    # print(pointx, pointy)
                    state[self.vert_size - pointy - 1, pointx] = -2

        for pointx in [self.agent_state[0] -1, self.agent_state[0], self.agent_state[0] + 1]:
            for pointy in [self.agent_state[1] -1, self.agent_state[1], self.agent_state[1] + 1]:
                if pointx not in [-1, self.horz_size] and pointy not in [-1, self.vert_size]:
                    state[self.vert_size - pointy - 1, pointx] = 2
        return state

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reset(self):

        """
        Reset the environemnt
        """

        self.agent_state = self.start_point[0], self.start_point[1]
        state = self.get_state()
        return state

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
