#!/usr/bin/env pybricks-micropython
"""Entry point for the three-axis robot arm controller on LEGO MINDSTORMS EV3.

Wires the four motors and two touch sensors to the EV3 brick, instantiates
the high-level ``ThreeAxisRobotArm`` controller, and dispatches the
top-level event-driven state machine via ``ThreeAxisRobotArm.run``. The
state machine alternates between SLEEP (motors coasted, LED red) and
ACTIVE (interactive ``xyz_control`` mode) on the brick's CENTER button;
see ``ThreeAxisRobotArm.run`` and ``ThreeAxisRobotArm.xyz_control`` for
full semantics, including the TOLERANCE_PAUSE sub-state.

The external gear trains for the four motors are exposed below as
module-level constants so that the physical mechanism behind each motor
port is documented at the call site. Pybricks scales every motor command
and encoder reading to the output (joint) shaft of the declared gear
train, so that the angle conventions in ``three_axis_robot_arm.py``
reason in terms of physical joint angles rather than motor-shaft angles.

References:
    Pybricks ev3devices.Motor -- ``docs/ev3-reference/api/motor.md``.
"""

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, TouchSensor
from pybricks.parameters import Port, Direction

from three_axis_robot_arm import ThreeAxisRobotArm


# =============================================================================
# External gear trains
#
# Each motor drives its joint through an external LEGO Technic gear train
# that Pybricks must be informed of, in order to scale ``run_target``,
# ``angle()``, ``run_to_angle`` and similar primitives to the output shaft.
# The convention (per the Pybricks documentation) is that a stage
# ``[A, B]`` denotes a motor-side pinion of A teeth meshed with a load-
# side wheel of B teeth, so that the output rotates by A/B for every
# motor rotation. Multiple stages are listed sequentially and multiply.
# =============================================================================

# Gripper: a single 1:24 reduction stage modelling the worm-like internal
# drive that converts motor rotation into jaw motion. One full motor
# rotation closes (or opens) the jaws by 360 / 24 = 15 degrees.
GRIPPER_GEAR_TRAIN = [1, 24]

# Base (yaw): two-stage train. The first stage is a 36-tooth pinion
# driving a 12-tooth wheel (a 3:1 increaser used internally for routing).
# The second stage is a nominal 1:56 reduction that captures the
# 56-tooth Technic turntable (LEGO part 4185), which carries the upper
# structure. The net ratio is 36/12 * 1/56 = 3/56, i.e. the turntable
# rotates 1 degree for every 18.67 degrees of motor rotation.
BASE_GEAR_TRAIN = [[36, 12], [1, 56]]

# Shoulder and elbow: identical 8-tooth pinion driving a 40-tooth wheel
# (a 1:5 reduction). One motor rotation produces 360 / 5 = 72 degrees of
# joint rotation, comfortably above the per-joint operating ranges.
SHOULDER_GEAR_TRAIN = [8, 40]
ELBOW_GEAR_TRAIN = [8, 40]


# =============================================================================
# Interactive session defaults
# =============================================================================

# Initial Cartesian target presented to ``xyz_control`` on every ACTIVE
# entry, expressed in the base frame (origin at joint 1, +x forward,
# +y leftward, +z upward). Chosen as a comfortable forward pose well
# within the workspace; the user can subsequently move the target with
# the brick buttons.
INITIAL_TARGET_X_CM = 25
INITIAL_TARGET_Y_CM = 0
INITIAL_TARGET_Z_CM = 0

# Cartesian step applied per UP / DOWN button event in ``xyz_control``.
# Smaller values yield finer control at the cost of perceived speed at
# the 20 Hz interactive loop frequency; empirically 0.5 cm offers a
# good balance of responsiveness and motion smoothness.
XYZ_STEP_CM = 0.5


def main():

    ev3_brick = EV3Brick()

    motor_gripper = Motor(Port.C, Direction.COUNTERCLOCKWISE, GRIPPER_GEAR_TRAIN)
    motor_first_axis = Motor(Port.A, Direction.COUNTERCLOCKWISE, BASE_GEAR_TRAIN)
    motor_second_axis = Motor(Port.B, Direction.CLOCKWISE, SHOULDER_GEAR_TRAIN)
    motor_third_axis = Motor(Port.D, Direction.COUNTERCLOCKWISE, ELBOW_GEAR_TRAIN)

    touch_sensor_right = TouchSensor(Port.S1)
    touch_sensor_left = TouchSensor(Port.S4)

    three_axis_robot_arm = ThreeAxisRobotArm(
        ev3_brick,
        motor_gripper,
        motor_first_axis,
        motor_second_axis,
        motor_third_axis,
        touch_sensor_right,
        touch_sensor_left,
        debug = True,
    )

    three_axis_robot_arm.run(
        x_default = INITIAL_TARGET_X_CM,
        y_default = INITIAL_TARGET_Y_CM,
        z_default = INITIAL_TARGET_Z_CM,
        step_cm = XYZ_STEP_CM,
    )


if __name__ == "__main__":

    main()
