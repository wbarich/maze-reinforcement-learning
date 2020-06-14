#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import random
import numpy as np
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class agent:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, start_horz, start_vert):

        self.pos_horz = start_horz
        self.pos_vert = start_vert
        self.pos_rot = 0

        self.agent_horz_size = 1
        self.agent_vert_size = 1

        self.previous_action = [0, 0, 0]
        self.momentum = 1

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def act(self, state = None):

        """
        Given the current state choose an action.

        action = [x_vel, y_vel, z_rot]
        """

        action_range = 10
        action = [random.randint(-action_range, action_range), random.randint(-action_range, action_range), 0]
        # action = [(self.momentum * self.previous_action[0]) + action[0], (self.momentum * self.previous_action[1]) + action[1], (self.momentum * self.previous_action[2]) + action[2]]
        self.previous_action = action

        return action

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
