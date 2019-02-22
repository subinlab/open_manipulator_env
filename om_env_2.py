#! /usr/bin/env python

import om_env_1


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(om_env_1.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, current_distance, previous_distance, is_reached, has_object
        distance_threshold, reward_type
        ):

        self.current_distance = 1
        self.previous_distance = 1
        self.is_reached = False

        self.has_object = has_object
        self.distance_threshold = distance_threshold
        self.reward_type = 'sparse'

        super(FetchEnv, self).__init__(n_actions=3)


    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    # RobotEnv methods
    # ----------------------------

    def _get_obs(self):
        """Returns the observation.
        """
        # _get_joint_obs: observation, achieved_goal
        # _get_target_obj_obs: desired_goal

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
        }

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        self._set_velocity_action()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        desired_goal = self._get_target_obj_obs()

        return desired_goal.copy()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        if self.is_dagger:
            print ('All demo trajectories are collected for this EPISODE')
            rospy.set_param('dagger_reset',"true") # param_name, param_value        
            print ('Waiting for new episode to start')        
            while not rospy.is_shutdown():
                if rospy.has_param('epi_start'):
                    break                    
            rospy.delete_param('epi_start')   
            print ('Now starts new eisode')
        else:
            rospy.set_param('ddpg_reset',"true") # param_name, param_value
            print ('Reset param published')
            print ('Now moves to start position')
            # _color_obs = self.getColor_observation()
            resetMsg = Bool()
            self.resetPub.publish(resetMsg)
            while not rospy.is_shutdown():
                if self.isReset:
                    self.isReset = False
                    break
            print ('Now starts new eisode')

    def check_for_termination(self):
        """Termination triggers done=True
        """
        X_RANGE = range(0.3,0.75)
        Y_RANGE = range(-0.5,0.5)
        Z_RANGE = range(-0.2,0.55)

        if not self.position[0] in X_RANGE or not self.position[1] in Y_RANGE or not self.position[2] in Z_RANGE:
            self.terminateCount +=1

        if self.terminateCount == 50:
            self.terminateCount =0
            return True
        else:
            return False

    def check_for_success(self):
        """Success triggers done=True
        """
        curDist = self.getDist()
        if curDist < self.distance_threshold:
            self.successCount +=1
            self.reward +=1
        if self.successCount == 50:
            self.successCount =0
            return True
        else:
            return False
