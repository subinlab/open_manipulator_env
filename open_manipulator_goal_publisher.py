#!/usr/bin/env python
# ROS Imports
import rospy
import geometry_msgs.msg


class GoalPublisher(object):
    def __init__(self):
        rospy.init_node('goal_publisher')

        self.goal_pub = rospy.Publisher('/teacher/ik_vel/', geometry_msgs.msg.Pose, queue_size=3)

    def goal_publisher(self):

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.goal_pose = geometry_msgs.msg.Pose()
            self.goal_pose.position.x = 0.29
            self.goal_pose.position.y = 0.0
            self.goal_pose.position.z = 0.20305    
            self.goal_pub.publish(self.goal_pose)
            rate.sleep()

        return self.goal_pose


def main():
    gp = GoalPublisher()
    gp.goal_publisher()

        
if __name__ == '__main__':
    try:    
        main()
    except rospy.ROSInterruptException:
                pass
