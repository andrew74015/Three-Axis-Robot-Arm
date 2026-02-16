
from pybricks.parameters import Stop, Color
from pybricks.tools import wait

from math import sqrt, degrees, acos, atan2


class ThreeAxisRobotArm:

    def __init__(self, ev3_brick, motor_gripper, motor_first_axis, motor_second_axis, motor_third_axis, touch_sensor_right, touch_sensor_left):

        self.ev3_brick = ev3_brick

        self.motor = [
            Motor(motor_gripper, 2 * 360, 0 * 24, 90 * 24, 0 * 24),
            Motor(motor_first_axis, 2 * 360, 0 * 56, 60 * 56, -60 * 56),
            Motor(motor_second_axis, 30, 2, 90, 30),
            Motor(motor_third_axis, 30, 90, 180, 60)
        ]

        self.touch_sensor_right = touch_sensor_right
        self.touch_sensor_left = touch_sensor_left

        self.powered = False
        self.ev3_brick.light.on(Color.RED)

        self.systemXYZ = SystemXYZ(30 * 0.8, 24 * 0.8)


    def toggle_powered(self):

        if self.powered:

            self.powered = False
            self.ev3_brick.light.on(Color.RED)

        else:

            self.powered = True
            self.ev3_brick.light.on(Color.YELLOW)

        for motor in self.motor:
            
            if self.powered:

                motor.reset_angle(motor.angle_startup)

            motor.run_angle(motor.angle_startup)

            if motor.get_angle() > motor.angle_max and self.powered:

                motor.run_angle(motor.angle_max)

            elif motor.get_angle() < motor.angle_min and self.powered:

                motor.run_angle(motor.angle_min)

            if motor.hold != self.powered:

                motor.toggle_hold()


    def debug_motor(self, id, delay = True, limit = True, reset = False, startup = 0):

        if not self.powered:

            return

        if id == 0:

            gear = 24

        elif id == 1:

            gear = 56
            
        elif id == 2 or id == 3:

            gear = 1

        angle = self.motor[id].get_angle()

        if not reset:

            startup = angle

        while not self.touch_sensor_right.pressed() or not self.touch_sensor_left.pressed():

            if self.touch_sensor_right.pressed():

                angle = angle + 1 * gear

            elif self.touch_sensor_left.pressed():

                angle = angle - 1 * gear

            if angle > self.motor[id].angle_max and limit:

                angle = self.motor[id].angle_max

            elif angle < self.motor[id].angle_min and limit:

                angle = self.motor[id].angle_min

            self.motor[id].run_angle(angle, delay)
            print("Debug motor", id, round(self.motor[id].get_angle() / gear))

            wait(10)

        if reset:

            self.motor[id].angle_startup = startup
            self.motor[id].reset_angle(self.motor[id].angle_startup)
            
        else:

            self.motor[id].run_angle(startup)


    def go_destination(self, x, y, z, grab = True):

        if not self.powered or self.systemXYZ.l1 + self.systemXYZ.l2 < sqrt(x * x + y * y):

            return

        self.systemXYZ.set_destination(x, y, z)

        for id, motor in enumerate(self.motor):

            if id == 0:

                continue

            angle = self.systemXYZ.get_angle(id)

            if angle < motor.angle_min or angle > motor.angle_max:

                return 

            motor.run_angle(angle, False)

        angle = self.motor[0].get_angle()

        while (not self.touch_sensor_right.pressed() or not self.touch_sensor_left.pressed()) and grab:

            if self.touch_sensor_right.pressed():

                angle = angle + 1 * 24

            elif self.touch_sensor_left.pressed():

                angle = angle - 1 * 24

            if angle > self.motor[0].angle_max:

                angle = self.motor[0].angle_max

            elif angle < self.motor[0].angle_min:

                angle = self.motor[0].angle_min

            self.motor[0].run_angle(angle, False)

            wait(10)
            

class Motor:

    def __init__(self, device, angle_speed, angle_startup, angle_max, angle_min):

        self.device = device
        self.angle_speed = angle_speed
        self.angle_startup = angle_startup
        self.angle_max = angle_max
        self.angle_min = angle_min
        self.hold = False


    def toggle_hold(self):

        if self.hold:

            self.hold = False
            self.device.stop()

        else:

            self.hold = True
            self.device.hold()

    
    def get_angle(self):

        return self.device.angle()


    def reset_angle(self, angle):

        self.device.reset_angle(angle)

    
    def run_angle(self, angle, delay = True):

        self.device.run_target(self.angle_speed, angle, wait = delay)


class SystemXYZ:

    def __init__(self, l1, l2):

        self.point = Point()
        self.l1 = l1
        self.l2 = l2
    

    def set_destination(self, x, y, z):

        self.point.x = x
        self.point.y = y
        self.point.z = z


    def get_angle(self, id):

        if (self.point.x, self.point.y, self.point.z) == (0, 0, 0):

            return

        if id == 1:

            angle = degrees(atan2(self.point.z, self.point.x))

            return round(angle) * 56

        elif id == 2:

            ip = sqrt(self.point.x * self.point.x + self.point.y * self.point.y)

            angle1 = degrees(atan2(self.point.y, self.point.x))

            angle2 = degrees(acos((self.l1 * self.l1 + ip * ip - self.l2 * self.l2)/(2 * self.l1 * ip)))

            return round(angle1 + angle2)

        elif id == 3:

            ip = sqrt(self.point.x * self.point.x + self.point.y * self.point.y)

            angle = degrees(acos((self.l1 * self.l1 + self.l2 * self.l2 - ip * ip)/(2 * self.l1 * self.l2)))

            return round(angle)


class Point:

    def __init__(self, x = 0, y = 0, z = 0):

        self.x = x
        self.y = y
        self.z = z