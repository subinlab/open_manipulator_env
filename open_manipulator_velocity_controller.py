#!/usr/bin/env python
#-----------------------------------------------------------------------
# Stripped down version of velocity_controller.py
# Runs at 100Hz
#
# Tasks:
# 1. Finds Jb and Vb
# 2. Uses naive least-squares damping to find qdot
#   (Addresses inf vel at singularities)
# 3. Publishes qdot to Sawyer using the limb interface
#
# Adopted from Chantiee Lee's code
#-----------------------------------------------------------------------
# Python Imports
import numpy as np
from math import pow, pi, sqrt
import tf.transformations as tr
import threading

# ROS Imports
import rospy
from std_msgs.msg import Bool, Int32, Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from urdf_parser_py.urdf import URDF

# Local Imports
import open_manipulator_description as s
import modern_robotics as r
import custom_logging as cl

####################
# GLOBAL VARIABLES #
####################
TIME_LIMIT = 7    #15s
DAMPING = 0.00 #0.06
JOINT_VEL_LIMIT = 2    #2rad/s

class VelocityController(object):
    def __init__(self):
        rospy.loginfo("Creating VelocityController class")

        # Grab M0 and Blist from saywer_MR_description.py
        self.M0 = s.M #Zero config of right_hand
        self.Blist = s.Blist #6x7 screw axes mx of right arm
        
        # Shared variables
        self.mutex = threading.Lock()
        self.time_limit = rospy.get_param("~time_limit", TIME_LIMIT)
        self.damping = rospy.get_param("~damping", DAMPING)
        self.joint_vel_limit = rospy.get_param("~joint_vel_limit", JOINT_VEL_LIMIT)
        self.q = np.zeros(4)        # Joint angles
        self.qdot = np.zeros(4)     # Joint velocities
        self.T_goal = r.FKinBody(self.M0, self.Blist, self.q[0:3])
        #self.T_goal = np.array(self.kin.forward(self.q))    # Ref se3
        
        self.isCalcOK = False
        # Subscriber
        self.joint_states_sub = rospy.Subscriber('/open_manipulator/joint_states', JointState, self.joint_states_cb)
        self.ref_pose_sub = rospy.Subscriber('/teacher/ik_vel/', Pose, self.ref_pose_cb)
        # Command publisher
        self.j1_pos_command_pub = rospy.Publisher('/open_manipulator/joint1_position/command', Float64, queue_size=3)
        self.j2_pos_command_pub = rospy.Publisher('/open_manipulator/joint2_position/command', Float64, queue_size=3)
        self.j3_pos_command_pub = rospy.Publisher('/open_manipulator/joint3_position/command', Float64, queue_size=3)
        self.j4_pos_command_pub = rospy.Publisher('/open_manipulator/joint4_position/command', Float64, queue_size=3)

        # self.vel_command_list_pub = rospy.Publisher('/vel/command/', Pose, queue_size = 3)


        # # rospy.wait_for_message("/ddpg/reset/",Float64)
        self.r = rospy.Rate(100)
        while not rospy.is_shutdown():
            self.calc_joint_vel()
            self.r.sleep()

    def joint_states_cb(self, joint_states):
        i=0
        while i<4:
            self.q[i] = joint_states.position[i+2]
            self.qdot[i] = joint_states.velocity[i+2]
            i += 1

    def ref_pose_cb(self, some_pose): # Takes target pose, returns ref se3
        #rospy.loginfo("Reference pose subscribed")
        p = np.array([some_pose.position.x, some_pose.position.y, some_pose.position.z])
        quat = np.array([some_pose.orientation.x, some_pose.orientation.y, some_pose.orientation.z, some_pose.orientation.w])
        goal_tmp = tr.compose_matrix(angles=tr.euler_from_quaternion(quat, 'sxyz'), translate=p)
        
        with self.mutex:
            goal_tmp = r.FKinBody(self.M0, self.Blist, self.q[0:3])
        goal_tmp[0:3,3] = p
        with self.mutex:
            self.T_goal = goal_tmp
        # self.joint_com_published = True
        return self.T_goal

    def get_phi(self, R_d, R_cur):
        phi = 0.5 * (np.cross(R_d[0:3, 0], R_cur[0:3, 0]) + np.cross(R_d[0:3, 1], R_cur[0:3, 1]) + np.cross(R_d[0:3, 2], R_cur[0:3, 2]))
        return phi

    def calc_joint_vel(self):

        rospy.logdebug("Calculating joint velocities...")

        # Body stuff
        Tbs = self.M0
        Blist = self.Blist

        # Desired config: base to desired - Tbd
        with self.mutex:
            T_sd = self.T_goal
            q_now = self.q

        T_cur = r.FKinBody(Tbs, Blist, q_now)
        print(T_cur)
        # Find transform from current pose to desired pose, error
        e = np.dot(r.TransInv(T_cur), T_sd)
        
        e = np.zeros(6)
        e[0:3] = self.get_phi(T_sd[0:3,0:3],T_cur[0:3,0:3])
        e[3:6] = T_sd[0:3,3] - T_cur[0:3,3]
        # Construct BODY JACOBIAN for current config
        Jb = r.JacobianBody(Blist, q_now) #6x7 mx
        Jv = Jb[3:6,:]

        # Desired ang vel - Eq 5 from Chiaverini & Siciliano, 1994
        # Managing singularities: naive least-squares damping
        invterm = np.linalg.inv(np.dot(Jv,Jv.T))
        qdot_new = np.dot(np.dot(Jv.T, invterm),e[3:6])
        #invterm = np.linalg.inv(np.dot(Jb.T,Jb))
        #qdot_new = np.dot(np.dot(invterm, Jb.T),e)
        
        # Scaling joint velocity
        minus_v = abs(np.amin(qdot_new))
        plus_v = abs(np.amax(qdot_new))
        if minus_v > plus_v:
            scale = minus_v
        else:
            scale = plus_v
        if scale > self.joint_vel_limit:
            qdot_new = 2.0*(qdot_new/scale)*self.joint_vel_limit
            # qdot_new = 1.0*(qdot_new/scale)*self.joint_vel_limit
        self.qdot = qdot_new #1x7
        
        dt = 0.001
        q1_desired = q_now[0] + 5*qdot_new[0] * dt
        q2_desired = q_now[1] + 5*qdot_new[1] * dt
        q3_desired = q_now[2] + 5*qdot_new[2] * dt
        q4_desired = q_now[3] + 5*qdot_new[3] * dt
        
        self.j1_pos_command_pub.publish(q1_desired)
        self.j2_pos_command_pub.publish(q2_desired)
        self.j3_pos_command_pub.publish(q3_desired)
        self.j4_pos_command_pub.publish(q4_desired)


        # Constructing dictionary
        #qdot_output = dict(zip(self.names, self.qdot))

        # Setting Sawyer right arm joint velocities
        #self.arm.set_joint_velocities(qdot_output)
        # print qdot_output
        return

def main():
    rospy.init_node('velocity_control')

    try:
        vc = VelocityControl()
        # rospy.wait_for_message("/ddpg/reset/",Float64)
        # r = rospy.Rate(100)
        # while not rospy.is_shutdown():
        #     vc.calc_joint_vel()
        #     r.sleep()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()

if __name__ == '__main__':
    main()
