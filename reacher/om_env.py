#! /usr/bin/env python

import rospy

import sys
import os
import time
import numpy as np
import math
import copy

import cv2
from cv_bridge import CvBridge, CvBridgeError
from string import Template

# reads open manipulator's state
from std_msgs.msg import *
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import ContactState
from tf import TransformListener

from ddpg.msg import GoalObs
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from gazebo_msgs.srv import (
    GetModelState
)
from get_model_gazebo_pose import GazeboModel
base_dir = os.path.dirname(os.path.realpath(__file__))

fixed_orientation = Quaternion(
                         x=-0.00142460053167,
                         y=0.999994209902,
                         z=-0.00177030764765,
                         w=0.00253311793936)

# to read ee position
from open_manipulator_msgs.msg import *



# gym
import gym
from gym import utils
from gym import error, spaces
from gym.utils import seeding
from gym.envs.registration import register


# Global variables
# -------------------------
ACTION_DIM = 3 # Cartesian
OBS_DIM = (100,100,3)      # POMDP
STATE_DIM = 24        # MDP


register(
        id='OMReach-v0',
        entry_point='openai_ros:task_envs.om_reach.om_reach.OMReachEnv',
        timestep_limit=1000,
    )


class OpenManipulatorEnv(gym.GoalEnv):
    def __init__(self, max_steps=700, isdagger=False, isPOMDP=False, train_indicator=0):
        """An implementation of OpenAI-Gym style robot reacher environment
        TODO: add method that receives target object's pose as state
        """        
        rospy.init_node('OpenManipulatorEnv')

        # for compatiability
        self.action_space = spaces.Box(-1., 1., shape=(ACTION_DIM,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        self.train_indicator = train_indicator # 0: Train 1:Test
        self.isdagger = isdagger
        self.isPOMDP = isPOMDP
        self.currentDist = 1
        self.previousDist = 1
        self.reached = False
        self.tf = TransformListener()
        self.bridge = CvBridge()
        self.max_steps = max_steps
        self.done = False
        self.reward = 0
        self.reward_rescale = 1.0
        self.isDemo = False
        self.reward_type = 'sparse'

        # instantiate dynamixel controller
        # try:
        #     self.dxl_pos = DynamixelPositionControl()
        #     rospy.loginfo("Dynamixel position controller has been instantiated")
        # except rospy.ROSInterruptException:
        #     pass
        # self.joint_command = JointCommand()
        # self.gripper =  intera_interface.Gripper("right")


        # Publihser nodes
        # -------------------------
        self.pub_gripper_position = rospy.Publisher('/open_manipulator/gripper_position/command', Float64, queue_size=1)
        self.pub_gripper_sub_position = rospy.Publisher('/open_manipulator/gripper_sub_position/command', Float64, queue_size=1)
        self.pub_joint1_position = rospy.Publisher('/open_manipulator/joint1_position/command', Float64, queue_size=1)
        self.pub_joint2_position = rospy.Publisher('/open_manipulator/joint2_position/command', Float64, queue_size=1)
        self.pub_joint3_position = rospy.Publisher('/open_manipulator/joint3_position/command', Float64, queue_size=1)
        self.pub_joint4_position = rospy.Publisher('/open_manipulator/joint4_position/command', Float64, queue_size=1)
        
        # TODO: manage this attribute when it's real test environment
        self.joints_position_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.kinematics_cmd = [0.0, 0.0, 0.0]

        # Subscriber nodes
        # -------------------------
        self.sub_joint_state = rospy.Subscriber('/open_manipulator/joint_states', JointState, self.joint_state_callback)
        # joint position/velocity/effort
        self.sub_kinematics_pose = rospy.Subscriber('/open_manipulator/gripper/kinematics_pose', KinematicsPose, self.kinematics_pose_callback)
        self.sub_robot_state = rospy.Subscriber('/open_manipulator/states', OpenManipulatorEnv, self.robot_state_callback)
        
        # cs position / orientation
        # variables for subscribe the joint states
        self.joint_names = ["gripper", "gripper_sub", "joint1", "joint2", "joint3", "joint4"]  # name: [gripper, gripper_sub, joint1, joint2, joint3, joint4]
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ee pose of robot -> used to compute reward.
        self.gripper_position = KinematicsPose().pose.position  # [x, y, z] cartesian position
        self.gripper_orientiation = KinematicsPose().pose.orientation  # [x, y, z, w] quaternion orientation

        #open manipulator statets
        self.moving_state = ""  # "MOVING" / "STOPPED
        self.actuator_state = ""  # "ACTUATOR_ENABLE" / "ACTUATOR_DISABLE" 

        # used for per-step elapsed time measurement
        self.tic = 0.0
        self.toc = 0.0
        self.elapsed = 0.0  
        self.initial_state = self.get_joints_states().copy()
        self._action_scale = 1.0


    # callback function
    # ----------------------------
    
    def joint_state_callback(self, msg):
        """Callback function of joint states subscriber.

        Argument: msg
        """
        self.joint_names = msg.name
        self.joint_positions = msg.position
        self.joint_velocities = msg.velocity
        self.joint_efforts = msg.effort
        # penalize jerky motion in reward for shaped reward setting.
        self.squared_sum_vel = np.linalg.norm(np.array(self.joint_velocities))

    def kinematics_pose_callback(self, msg):
        """Callback function of gripper kinematic pose subscriber.

        Argument: msg
        """        
        self.kinematics_pose = msg
        self.gripper_position = self.kinematics_pose.pose.position
        self.gripper_orientiation = self.kinematics_pose.pose.orientation

    def robot_state_callback(self, msg):
        self.moving_state = msg.open_manipulator_moving_state # "MOVING" / "STOPPED"
        self.actuator_state = msg.open_manipulator_actuator_state # "ACTUATOR_ENABLE" / "ACTUATOR_DISABLE"        

    # get and set function
    # ----------------------------

    def get_joints_states(self):
        """Returns current joints states of robot including position, velocity, effort

        Returns: Float64[] self.joints_position, self.joints_velocity, self.joint_effort
        """
        return self.joint_positions, self.joint_velocities, self.joint_efforts


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
        self.set_joints_position(act)
        if not self.is_real:
            self.reward = self.compute_reward()
            if self.check_for_termination():
                print ('======================================================')
                print ('Terminates Episode current Episode : OUT OF BOUNDARY')
                print ('======================================================')
                self.done = True
        _joint_Pos, _joint_vels, _joint_effos = self.get_joints_states()
        # obj_pos = self._get_target_obj_obs() # TODO: implement this function call.                       

        if np.mod(step, 10)==0:
            if not is_real:
                print("DISTANCE : ", curDist)
            print("PER STEP ELAPSED : ", self.elapsed)
            print("SPARSE REWARD : ", self.reward_rescale * self.reward)
            print("Current EE pos: ", self.gripper_position)
            print("Actions: ", act)

        obs = np.array([_joint_pos, _joint_vels, _joint_effos])

        return obs, self.reward_rescale * self.reward, self.done


    def _reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_gazebo_world()
        obs = self._get_obs()
        return obs


    def _check_robot_moving(self):
        """Check if robot has reached its initial pose.
        """
        while not rospy.is_shutdown():
            if self.moving_state=="STOPPED":
                break
        return True


    def _reset_gazebo_world(self):
        """
        Method that randomly initialize the state of robot agent and surrounding envs (including target obj.)
        """
        self.pub_gripper_position.publish(np.random.uniform(0.0, 0.1)) 
        self.pub_joint1_position.publish(np.random.uniform(-0.1, 0.1)) 
        self.pub_joint2_position.publish(np.random.uniform(-0.1, 0.1)) 
        self.pub_joint3_position.publish(np.random.uniform(-0.1, 0.1)) 
        self.pub_joint4_position.publish(np.random.uniform(-0.1, 0.1)) 
        _load_target_block()
        return _check_robot_moving()


    def _load_target_block(block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
                        block_reference_frame="world"):
        # Get Models' Path
        model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
        
        # Load Block URDF
        block_xml = ''

        num = randint(1,3)
        num = str(num)
        with open (model_path + "block/model_"+num+".urdf", "r") as block_file:
            block_xml=block_file.read().replace('\n', '')
        
        rand_pose = Pose(position=Point(x=np.random.uniform(0.45,0.63), 
                                        y=np.random.uniform(0.45,0.63), 
                                        z=np.random.uniform(0.45,0.63)),
                    orientation=overhead_orientation)   
        
        # Spawn Block URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            resp_urdf = spawn_urdf("block", block_xml, "/",
                                rand_pose, block_reference_frame)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def _check_for_termination(self):
        """
        Check if the agent has reached undesirable state. If so, terminate the episode early. 
        """
        raise NotImplementedError()

    def _check_for_success(self):
        """
        Check if the agent has reached undesirable state. If so, terminate the episode early. 
        """
        raise NotImplementedError()


    def _compute_reward(self):
        """Computes shaped/sparse reward for each episode.
        """
        cur_dist = self._get_dist()
        if self.reward_type == 'sparse':
            return (cur_dist <= self.distance_threshold).astype(np.float32) # 1 for success else 0
        else:
            return -cur_dist -self.squared_sum_vel # -L2 distance -l2_norm(joint_vels)

    def _get_dist(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            object_state_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            object_state = object_state_srv("block", "world")
            self._obj_pose = np.array([object_state.pose.position.x, object_state.pose.position.y, object_state.pose.position.z + DIST_OFFSET])
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))        
        _ee_pose = np.array(self.gripper_position) # FK state of robot
        return np.linalg.norm(_ee_pose-self._obj_pose)

    def close(self):        
        rospy.signal_shutdown("done")
