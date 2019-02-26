#! /usr/bin/env python

import rospy
import time
import numpy as np
import math
import copy
import numpy

# messages
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from open_manipulator_msgs.msg import KinematicsPose

# our controller
from open_manipulator_velocity_controller import VelocityController
from open_manipulator_goal_publisher import GoalPublisher

# tf
from tf.transformations import euler_from_quaternion

# gazebo
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection

# gym
import gym
from gym import utils
from gym import error, spaces
from gym.utils import seeding
from gym.envs.registration import register


# Global variables
# -------------------------
ACTION_DIM = 3  # Cartesian


register(
        id='FetchReach-v0',
        entry_point='openai_ros:task_envs.fetch_reach.fetch_reach.FetchReachEnv',
        timestep_limit=1000,
    )


class RobotEnv(gym.GoalEnv):
    def __init__(self, max_steps=700, ACTION_DIM, train_indicator=0):
        rospy.init_node('gym_environment/open_manipulator')

        self.train_indicator = train_indicator # 0: Train 1:Test

         # Publihser nodes
        # -------------------------
        self.pub_gripper_position = rospy.Publisher('/open_manipulator/gripper_position/command', Float64, queue_size=1)
        self.pub_gripper_sub_position = rospy.Publisher('/open_manipulator/gripper_sub_position/command', Float64, queue_size=1)
        self.pub_joint1_position = rospy.Publisher('/open_manipulator/joint1_position/command', Float64, queue_size=1)
        self.pub_joint2_position = rospy.Publisher('/open_manipulator/joint2_position/command', Float64, queue_size=1)
        self.pub_joint3_position = rospy.Publisher('/open_manipulator/joint3_position/command', Float64, queue_size=1)
        self.pub_joint4_position = rospy.Publisher('/open_manipulator/joint4_position/command', Float64, queue_size=1)
        
        # For publish joint position
        self.joints_position_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        # Subscriber nodes
        # -------------------------
        self.sub_joint_state = rospy.Subscriber('/open_manipulator/joint_states', JointState, self.joint_stateCB)
        self.sub_kinematics_pose = rospy.Subscriber('/open_manipulator/gripper/kinematics_pose', KinematicsPose, self.kinematics_poseCB)
        
        
        # For subscribe joint states
        self.joints_name = ["", "", "", "", "", ""]  # name: [gripper, gripper_sub, joint1, joint2, joint3, joint4]
        self.joints_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joints_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # For subscribe kinematics pose
        self.gripper_position = KinematicsPose().pose.position  # [x, y, z] cartesian position
        self.gripper_orientiation = KinematicsPose().pose.orientation  # [x, y, z, w] quaternion orientation



        # for compatiability
        self.action_space = spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype='float32')
        self.observation_space = spaces.Dict(dict(

            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))


        # env variables
        # -------------------------

        self.done = False
        self.successCount =0
        self.terminateCount =0
        self.reward = 0

        self.joints_position, self.joints_velocity, self.joints_effort = self._get_joint_obs()
        self.obj_position = self._get_target_obj_obs()        

        self.seed()
        self._env_setup()  # Initial configuration of the environment.

        self.resize_factor = 100/400.0
        self.resize_factor_real = 100/650.0

        self.current_distance = 1
        self.previous_distance = 1
        self.is_reached = False
        self.tf = TransformListener()

        self.max_steps = max_steps
        self.reward_rescale = 1.0
        self.is_demo = False
        self.reward_type = 'sparse'

        # used for per-step elapsed time measurement
        self.tic = 0.0
        self.toc = 0.0
        self.elapsed = 0.0

        self._action_scale = 1.0

    # callback function
    # ----------------------------
    
    def joint_state_callback(self, msg):
        """Callback function of joint states subscriber.

        Argument: msg
        """
        self.joints_states = msg

        self.joints_name = self.joints_states.name
        self.joints_position = self.joints_states.position
        self.joints_velocity = self.joints_states.velocity
        self.joints_effort = self.joints_states.effort


    def kinematics_pose_callback(self, msg):
        """Callback function of gripper kinematic pose subscriber.

        Argument: msg
        """        
        self.kinematics_pose = msg
        self.gripper_position = self.kinematics_pose.pose.position
        self.gripper_orientiation = self.kinematics_pose.pose.orientation


    # get and set function
    # ----------------------------

    def get_joints_states(self):
        """Returns current joints states of robot including position, velocity, effort

        Returns: Float64[] self.joints_position, self.joints_velocity, self.joint_effort
        """
        return self.joints_position, self.joints_velocity, self.joint_effort


    def get_gripper_pose(self):
        """Returns gripper end effector position

        Returns: Pose().position, Pose().orientation
        """      
        return self.gripper_position, self.gripper_orientiation


    def get_gripper_position(self):
        """Returns gripper end effector position

        Returns: Pose().position
        """      
        return self.gripper_position


    def set_joints_position(self, joints_angles):
        """Move joints using joint position command publishers.
        
        Argument: joints_position_cmd
        self.joints_position_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        rate = rospy.Rate(10) # 10hz 
        while not rospy.is_shutdown():
            rospy.loginfo(self.joints_position)
            self.pub_gripper_position.publish(self.joints_position[0])
            self.pub_gripper_sub_position.publish(self.joints_position[1])
            self.pub_joint1_position.publish(self.joints_position[2])
            self.pub_joint2_position.publish(self.joints_position[3])
            self.pub_joint3_position.publish(self.joints_position[4])
            self.pub_joint4_position.publish(self.joints_position[5])            
            rate.sleep()


    # gazebo simulation function
    # ----------------------------
    def unpause_sim(self):
        self.gazebo.unpauseSim()


    def pause_sim(self):
        self.gazebo.pauseSim()           


    # gym Env methods
    # ----------------------------
    def _seed(self, seed=None): #overriden function
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    
    

    def _step(self, action):#overriden function
        """
        Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done
        """
        self.prev_tic = self.tic
        self.tic = time.time()
        self.elapsed =  time.time()-self.prev_tic
        self.done = False
        if step == self.max_steps:
            self.done = True

        act = action.flatten().tolist()
        self._set_action(act)
        if not self.is_real:
            self.reward = self.compute_reward()
            if self.check_for_termination():
                print ('======================================================')
                print ('Terminates Episode current Episode : OUT OF BOUNDARY')
                print ('======================================================')
                self.done = True
        joint_pos, joint_vels, joint_effos = self._get_joint_obs()
        obj_pos = self._get_target_obj_obs()                        

        if np.mod(step, 10)==0:
            if not isReal:
                print("DISTANCE : ", curDist)
            print("PER STEP ELAPSED : ", self.elapsed)
            print("SPARSE REWARD : ", self.reward_rescale * self.reward)
            print("Current EE pos: ", self.gripper_position)
            print("Actions: ", act)

        obs = [joint_pos, joint_vels, joint_effos]

        return obs, self.reward_rescale * self.reward, self.done


    def _reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        self.done = False
        self.successCount =0
        self.terminateCount =0
        self.reward = 0

        self.joints_position, self.joints_velocity, self.joints_effort = self._get_joint_obs()
        self.obj_position = self._get_target_obj_obs()

        obs = self.joints_position, self.joints_velocity, self.joints_effort

        return obs


    def _close(self):
        rospy.signal_shutdown("done")


    def _set_velocity_action(self, action):
        """Applies the given action to the simulation.
        OpenAI gym style function
        Set action to move joints.

        Arguments: action
        """
        self.vel_ctrl = VelocityController()
        self.vel_ctrl.ref_pose_cb(action)
        self.vel_ctrl.calc_joint_vel()


    def _set_cartesian_action(self, action):
        self.cartesian_command = action


    def _get_joint_obs(self):
        """Returns the joints states including position, velocity, effort.
        Using _get_joints_obs() instead of openai gym style function _get_obs().

        Returns: Float64 obs_joints_position, obs_joints_velocity, obs_joints_effort
        """
        self.obs_joints_position, self.obs_joints_velocity, self.obs_joints_effort  = self.get_current_joints_states()
        while not self.obs_joints_position:
            print('waiting joint vals')
            self.obs_joints_position, self.obs_joints_velocity, self.obs_joints_effort  = self.get_current_joints_states()

        return self.obs_joints_position, self.obs_joints_velocity, self.obs_joints_effort


    def _get_target_obj_obs(self):
        """Returns the target object pose. Experimentally supports only position info.

        Returns: Pose() destPos
        """
        self.goal_pub = GoalPublisher()
        self.destPos = self.goal_pub.goal_publisher

        return self.destPos


    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
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


    def compute_reward(self):
        """ Reward computation for non-goalEnv.
        """
        curDist = self.getDist()
        if self.reward_type == 'sparse':
            return (curDist <= self.distance_threshold).astype(np.float32) # 1 for success else 0
        else:
            return -curDist -self.squared_sum_vel # -L2 distance -l2_norm(joint_vels)
    

    def getDist(self):
        DIST_OFFSET = -0.9+0.025-0.0375
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            object_state = object_state_srv("block", "world")
            self.destPos = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z + DIST_OFFSET])
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))        
        self.position = self.get_gripper_position()
        currentPos = np.array((self.position[0],self.position[1],self.position[2]))        


    # GoalEnv methods
    # ----------------------------
    def compute_goal_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d


    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
