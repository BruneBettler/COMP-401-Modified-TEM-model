#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import torch

# Functions for accessing data that TEM trains on: sequences of [state,observation,action] tuples

class World:
    def __init__(self, graph_path, session_id, env_data):
        # Environment and walk data are stored in a .h5 file.

        self.n_locations = 9

        self.session_id = session_id
        self.session_start_index, self.session_end_index = self.get_session_indices(env_data)  # session_end_index = the index of the first trial in the following session (use this value for in range)

        self.session_observation_images = env_data['observation_image'][(5 * self.session_start_index): (5 * self.session_end_index)]
        self.session_9XY_locations = env_data['9XY_condition'][self.session_start_index: self.session_end_index]
        self.session_actions = env_data['action'][self.session_start_index: self.session_end_index]

        self.num_steps = self.session_end_index - self.session_start_index

        '''# todo: remove which-one_hot this is for me to debug!
        if session_id == b'R20140910':
            self.oneHot = np.load("bR20140910_oneHot.npy")
        if session_id == b'R20140926':
            self.oneHot = np.load("bR20140926_oneHot.npy")'''

        self.oneHot = self.make_hot_object_array()


    def make_total_1hot_arrayfile(self):
        """returns an NxN matrix with each "pixel" location containing a 1-hot encoded representation of the type of object located there.
                        n_pixel_types = 13
                        0 = wall (2,5)
                        1 = empty path (accessible to agent) (1,0)
                        2 = orientation cue blue (7,2)
                        3 = orientation cue red (7,0)
                        4 = context cue yellow (13,4)
                        5 = context cue purple (13,3)
                        6 = purple object (accessible to agent) (6,3)
                        7 = yellow object (accessible to agent) (6,4)
                        8 = gray object (accessible to agent) (6,5)
                        9 = red object (accessible to agent) (6,0)
                        10 = blue object (accessible to agent) (6,2)
                        11 = green object (accessible to agent) (6,1)
                        12 = orange object  (accessible to agent) (6,6)
                """
        total_agent_view = self.session_observation_images[:].reshape((-1, self.session_observation_images.shape[1], self.session_observation_images.shape[1], self.session_observation_images.shape[2]))
        total_hot_agent_view = np.zeros((self.num_steps, 5, 5, 14))

        which_obj = [12, 1, 21, 7, 65, 52, 24, 30, 36, 6, 18, 12, 42, 48]

        for step in range(0, total_hot_agent_view.shape[0]):
            single_hot_agent_view = np.zeros((5, 5, 14))
            for hj in range(0, 5):
                for hi in range(0, 5):
                    # determine which of 14 labels corresponds to "pixel"
                    obj, color, _ = total_agent_view[step:step + 1][:][:][0][hj][hi]
                    objIndex = which_obj.index((obj * color) + obj)
                    single_hot_agent_view[hi][hj] = np.eye(14)[
                        objIndex]  # 1hot for the object found at the "pixel" location
            total_hot_agent_view[step] = single_hot_agent_view

        #np.save('bR20140926_oneHot.npy', total_hot_agent_view)

        return total_hot_agent_view

    def make_hot_object_array(self):
        """returns an NxN matrix with each "pixel" location containing a 1-hot encoded representation of the type of object located there.
                        n_pixel_types = 13
                        0 = wall (2,5)
                        1 = empty path (accessible to agent) (1,0)
                        2 = orientation cue blue (7,2)
                        3 = orientation cue red (7,0)
                        4 = context cue yellow (13,4)
                        5 = context cue purple (13,3)
                        6 = purple object (accessible to agent) (6,3)
                        7 = yellow object (accessible to agent) (6,4)
                        8 = gray object (accessible to agent) (6,5)
                        9 = red object (accessible to agent) (6,0)
                        10 = blue object (accessible to agent) (6,2)
                        11 = green object (accessible to agent) (6,1)
                        12 = orange object  (accessible to agent) (6,6)
                9 objects per session though so observation with 1-hot is 9x25 = 225 compressed into 4bytes each so 5x5x4
                """
        total_agent_view = self.session_observation_images[:].reshape((-1, self.session_observation_images.shape[1],
                                                                       self.session_observation_images.shape[1],
                                                                       self.session_observation_images.shape[2]))
        total_hot_agent_view = np.zeros((self.num_steps, 5, 5, 4))

        which_obj = [12, 1, 21, 7, 65, 52, 24, 30, 36, 6, 18, 12, 42]

        for step in range(0, total_hot_agent_view.shape[0]):
            single_hot_agent_view = np.zeros((5, 5, 4))
            for hj in range(0, 5):
                for hi in range(0, 5):
                    # determine which of 14 labels corresponds to "pixel"
                    obj, color, _ = total_agent_view[step:step + 1][:][:][0][hj][hi]
                    objIndex = which_obj.index((obj * color) + obj)
                    binary = format(objIndex, '04b')
                    binary_array = np.array(list(binary), dtype=int)
                    single_hot_agent_view[hi][hj] = binary_array  # 4 bit value for the object found at the "pixel" location
            total_hot_agent_view[step] = single_hot_agent_view

        # np.save('bR20140926_oneHot.npy', total_hot_agent_view)

        return total_hot_agent_view

    def generate_walks(self, walk_length=10, n_walk=100):
        # a walk is defined as a series of trials per session

        walks = []  # This is going to contain a list of (state, observation, action) tuples
        for currWalk in range(n_walk):
            new_walk = []
            new_walk = self.walk_default(new_walk, walk_length)
            walks.append(new_walk)
        return walks

    def walk_default(self, walk, walk_length):
        # Finish the provided walk until it contains walk_length steps
        for curr_step in range(walk_length - len(walk)):
            # Get new location based on previous action and location
            new_location = self.get_location(curr_step)                  # returns an int location of where agent is
            # Get new observation at new location
            new_observation = self.get_observation(curr_step)
            # Get new action based on policy at new location
            new_action = self.get_action(curr_step)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk

    def get_location(self, curr_step):
        # get the location from our env h5 file
        location = self.session_9XY_locations[curr_step]
        return location

    def get_observation(self, curr_step):
        # returns the 5,5,4 np array containing the onehot-at-each-gridspace
        tensor_observation = torch.from_numpy(self.oneHot[curr_step])
        tensor_observation_flat = torch.flatten(tensor_observation).float()
        return tensor_observation_flat

    def get_action(self, curr_step):                                       # todo: may be 'observation_action'?? think about which it would be !!!
        # returns the int representation of the current index action
        action = self.session_actions[curr_step]
        return action

    def get_session_IDs(self, s_index, e_index):
        # check the done list
        sessions = []
        for index in range(s_index,e_index+1):
            if self.env_data['done'][index] == True:
                info = [self.env_data['session_id'][index], index]  # [session_name, session_end_index]
                sessions.append(info)
        return sessions


    def get_session_indices(self, env_data):
        # find index at which session begins
        for currSession_id in range(len(env_data['session_id'])):
            if env_data['session_id'][currSession_id] == self.session_id:
                start = currSession_id
                break
        # find the index at which the session ends
        for done_index in range(len(env_data['done'][start:])): # TODO check edge case when at end of list of env
            if (done_index + start + 1) != len(env_data['done']):
                if env_data['done'][done_index + start + 1]:
                    end = done_index + start + 1
                    break
            else: end = len(env_data['done'])-1
        return start, end


if __name__ == '__main__':
   ''' graph_path = 'envs/5x3bone.jason'
    session_id = b'R20140926'
    h5_file_path = '/Users/brune/PycharmProjects/TEM_2D-Memory/activations_memory-cc-v2_rnn_3_3_MiniGrid-Associative-MemoryS7RMTM-v0.h5'
    hf = h5py.File(h5_file_path, 'r')
    env_data = hf['env_data']

    world0 = World(graph_path, session_id, env_data)
    world0.make_total_1hot_arrayfile()'''












