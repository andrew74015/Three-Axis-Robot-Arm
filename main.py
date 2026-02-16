#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, TouchSensor
from pybricks.parameters import Port, Direction
from pybricks.tools import wait

from three_axis_robot_arm import ThreeAxisRobotArm


def main():

    ev3_brick = EV3Brick()
    
    motor_gripper = Motor(Port.C, Direction.COUNTERCLOCKWISE)
    motor_first_axis = Motor(Port.A, Direction.CLOCKWISE, [36, 12])
    motor_second_axis = Motor(Port.B, Direction.CLOCKWISE, [8, 40])
    motor_third_axis = Motor(Port.D, Direction.CLOCKWISE, [8, 40])

    touch_sensor_right = TouchSensor(Port.S1)
    touch_sensor_left = TouchSensor(Port.S4)

    three_axis_robot_arm = ThreeAxisRobotArm(ev3_brick, motor_gripper, motor_first_axis, motor_second_axis, motor_third_axis, touch_sensor_right, touch_sensor_left)

    three_axis_robot_arm.toggle_powered()

    three_axis_robot_arm.go_destination(30, 20, 20, True)

    wait(1000)

    three_axis_robot_arm.go_destination(30, 20, -20, True)

    three_axis_robot_arm.toggle_powered()

if __name__ == "__main__":

    main()