
"""Three-axis robot arm controller for LEGO MINDSTORMS EV3.

This module implements analytical inverse kinematics for a 3-degree-of-freedom
RRR anthropomorphic manipulator equipped with a parallel-jaw gripper, on the
LEGO MINDSTORMS EV3 platform via Pybricks MicroPython.

Architecture
------------
    ThreeAxisRobotArm : High-level controller. Orchestrates power-on / power-off
        sequences, target validation, parallel motor dispatch, drift
        compensation, and the interactive xyz_control mode.
    Joint             : Thin wrapper around pybricks.ev3devices.Motor adding
        per-joint operating limits, a kinematic offset, and an at-target
        predicate aligned with the position tolerance.
    SystemXYZ         : Pure analytical inverse kinematics solver, independent
        of the EV3 hardware abstraction.

Hardware configuration
----------------------
    Four DC motors (gripper, base, shoulder, elbow), two touch sensors,
    one EV3 brick.

References
----------
    [1] Lynch, K. M. and Park, F. C. (2017). Modern Robotics: Mechanics,
        Planning, and Control. Cambridge University Press.
    [2] Pybricks Documentation, v2.0.
"""

from pybricks.parameters import Color, Button
from pybricks.tools import wait

from math import sqrt, degrees, acos, atan2


# =============================================================================
# Module-level constants
# =============================================================================

# -----------------------------------------------------------------------------
# Position tolerance and motion envelope (Pybricks Control class configuration)
# -----------------------------------------------------------------------------

# Angular tolerance below which a joint is considered to have reached its
# commanded target, in degrees. Used uniformly by Joint.at_target, by the
# embedded Pybricks position controller (target_tolerances), and by the
# post-blocking-move drift correction.
POSITION_TOLERANCE_DEG = 2

# Motion-profile acceleration limit applied to every joint controller,
# in deg / s^2. Lower values produce smoother motion at the cost of
# response time; higher values produce the converse.
MOTOR_ACCELERATION = 200

# Hard cap on PID controller speed for every joint, in deg / s on the
# output shaft. Configured at construction via ``control.limits(speed)``
# so the same envelope applies uniformly to every motion primitive
# (``run_target`` clamps its requested speed to this cap, and
# ``track_target`` uses it as its top speed since it does not accept
# a per-call speed argument). Set explicitly here rather than left
# at the (much higher) Pybricks default to keep retargeting in
# ``xyz_control`` predictable and consistent with the trapezoidal-
# profile blocking moves used elsewhere.
MOTOR_MAX_SPEED = 30

# Speed used for the power-on and power-off blocking sweeps, in deg / s
# on the output shaft. Slower than ``MOTOR_MAX_SPEED`` so that the
# parking and calibration moves are visibly gentle on the LEGO gear
# trains and during the SLEEP <-> ACTIVE state-machine transitions.
POWER_SEQUENCE_SPEED_DEG_S = 15

# -----------------------------------------------------------------------------
# Blocking-move wait timing
# -----------------------------------------------------------------------------

# Polling cadence for blocking move waits, in milliseconds. Chosen to
# balance responsiveness of the at-target detection against the cost
# of per-tick status printouts when running with debug=True.
WAIT_POLL_PERIOD_MS = 200

# Hard upper bound on the wall-clock time spent waiting for a blocking
# move to complete, in milliseconds. Provides a fail-safe for the worst
# case in which a joint is mechanically blocked but does not register
# as stalled.
WAIT_TIMEOUT_MS = 10000

# Maximum number of post-move drift-correction retry attempts per joint.
# After this many attempts the joint is logged as persistently drifting
# and the controller proceeds rather than blocking indefinitely.
DRIFT_CORRECTION_MAX_ATTEMPTS = 2

# -----------------------------------------------------------------------------
# Interactive xyz_control loop
# -----------------------------------------------------------------------------

# Tick period for the interactive xyz_control event loop, in milliseconds.
# A 50 ms tick (20 Hz) is fast enough for responsive UP / DOWN retargeting
# while leaving headroom for the per-tick IK validation and the tolerance
# check.
XYZ_TICK_PERIOD_MS = 50

# Number of consecutive ticks the CENTER button must be held in xyz_control
# to register as a long-press exit signal (XYZ_TICK_PERIOD_MS * 30 = 1.5 s).
XYZ_LONG_PRESS_TICKS = 30

# Throttling factor for status printouts during a sustained UP / DOWN hold:
# a status snapshot is printed every Nth tick (here, twice per second at
# the default tick period) plus one unconditional snapshot on release.
XYZ_PRINT_EVERY_N_TICKS = 10

# -----------------------------------------------------------------------------
# Gripper inner loop
# -----------------------------------------------------------------------------

# Tick period for the gripper inner loop driven by the touch sensors,
# in milliseconds. Faster than the main loop because the gripper is a
# fine-grained user input and should feel immediately responsive.
GRIPPER_TICK_PERIOD_MS = 10

# Throttling factor for status printouts inside the gripper loop when
# debug=True (otherwise the loop floods the console at GRIPPER_TICK_PERIOD_MS).
GRIPPER_DEBUG_PRINT_EVERY_N = 20

# Angular step applied to the gripper per tick of the inner loop,
# in degrees on the output (jaw) shaft.
GRIPPER_STEP_DEG = 1

# -----------------------------------------------------------------------------
# Touch sensor input
# -----------------------------------------------------------------------------

# Polling period used to debounce a touch-sensor release at the
# both-pressed-exit path of xyz_control, in milliseconds. Reused by
# ``_wait_button_release`` for brick-button debouncing.
TOUCH_DEBOUNCE_PERIOD_MS = 20

# -----------------------------------------------------------------------------
# Tolerance-pause UX threshold
# -----------------------------------------------------------------------------

# Threshold for triggering the visible "tolerance pause" sub-state of
# xyz_control: when any IK joint's encoder lags its target by more than
# this many degrees, the loop freezes UP / DOWN / LEFT / RIGHT input
# until the Pybricks PID controller closes the gap. Distinct from
# POSITION_TOLERANCE_DEG (the narrower at-target tolerance, shared with
# the Pybricks internal controller via target_tolerances).
TOLERANCE_PAUSE_THRESHOLD_DEG = 5


# =============================================================================
# Robot geometry
#
# Distances between joint axes, expressed in LEGO Technic units. One unit
# equals 0.8 cm (the standard 8-mm hole-to-hole pitch), so a Technic beam
# with N holes spans exactly (N - 1) units.
#
# This parametrization aligns directly with the Denavit-Hartenberg link
# parameters of a 3R anthropomorphic manipulator: SHOULDER_Z_OFFSET maps to
# the link offset d_1, while UPPER_ARM and FOREARM map to the link lengths
# a_2 and a_3 respectively.
# =============================================================================

# Length of one LEGO Technic module, in centimeters. Corresponds to the
# standard hole-to-hole pitch of Technic beams (8 mm). Used to convert
# the LEGO-unit measurements below into the centimeter inputs expected
# by the analytical IK solver.
LEGO_UNIT_CM = 0.8

# Vertical distance from the joint 1 axis (base, yaw) to the joint 2 axis
# (shoulder, pitch), measured along +z. Equivalent length: a 6-hole beam.
SHOULDER_Z_OFFSET_LEGO_UNITS = 5

# Length of the upper-arm link (l_1), between the joint 2 axis (shoulder)
# and the joint 3 axis (elbow). Equivalent length: a 31-hole beam.
UPPER_ARM_LEGO_UNITS = 30

# Length of the forearm link (l_2), between the joint 3 axis (elbow) and
# the gripper grip point. Equivalent length: a 25-hole beam.
FOREARM_LEGO_UNITS = 24


# =============================================================================
# Helper functions
# =============================================================================

def _clamp(value, lower, upper):
    """Constrain ``value`` to the inclusive interval ``[lower, upper]``.

    Args:
        value: The number to constrain.
        lower: Lower bound, inclusive.
        upper: Upper bound, inclusive.

    Returns:
        ``lower`` if ``value < lower``; ``upper`` if ``value > upper``;
        ``value`` otherwise.
    """

    return max(lower, min(value, upper))


class ThreeAxisRobotArm:

    # =========================================================================
    # Class constants
    # =========================================================================

    # Joint indices that participate in the inverse-kinematics chain (every
    # joint except the gripper). All three are dispatched together for any
    # valid Cartesian target, since validate-then-execute rejects targets
    # that any of them cannot reach.
    IK_CHAIN_JOINT_IDS = [1, 2, 3]

    # Joint indices in calibration and parking order: gripper, base, shoulder,
    # elbow. Bottom-up by kinematic chain, so that lower-axis encoder zeroing
    # happens before the resting joints above can be commanded; the order also
    # mirrors the natural mechanical assembly of the arm.
    POWER_SEQUENCE = [0, 1, 2, 3]


    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(self, ev3_brick, motor_gripper, motor_first_axis, motor_second_axis, motor_third_axis, touch_sensor_right, touch_sensor_left, debug = False):

        self.ev3_brick = ev3_brick
        self.debug = debug

        self.joints = [
            Joint(motor_gripper, 30, 0, 80, 0, "grip"),
            Joint(motor_first_axis, 30, 0, 60, -60, "base"),
            Joint(motor_second_axis, 30, 5, 90, 25, "shldr"),
            Joint(motor_third_axis, 30, 90, 120, 0, "elbow")
        ]

        self.touch_sensor_right = touch_sensor_right
        self.touch_sensor_left = touch_sensor_left

        self.powered = False
        self.ev3_brick.light.on(Color.RED)

        self.systemXYZ = SystemXYZ(
            l1 = UPPER_ARM_LEGO_UNITS * LEGO_UNIT_CM,
            l2 = FOREARM_LEGO_UNITS * LEGO_UNIT_CM,
            shoulder_z_offset = SHOULDER_Z_OFFSET_LEGO_UNITS * LEGO_UNIT_CM,
        )


    # =========================================================================
    # Top-level state machine
    # =========================================================================

    def run(self, x_default, y_default, z_default, step_cm):
        """Top-level interactive loop: alternate SLEEP and ACTIVE states.

        Replaces a static ``power_on -> xyz_control -> power_off``
        sequence with an event-driven state machine. The robot starts
        in SLEEP (motors coasted, LED red); a CENTER tap calibrates
        the encoders via ``power_on`` and immediately enters
        ``xyz_control``. Exiting xyz_control (CENTER hold or both
        touch sensors) parks the arm via ``power_off`` and returns
        the controller to SLEEP, ready for another session. The only
        way to leave this loop is the brick's BACK button, which
        hard-kills the program at the Pybricks-runtime level.

        Args:
            x_default: Initial forward coordinate, in centimeters,
                in the base frame.
            y_default: Initial leftward coordinate, in centimeters.
            z_default: Initial upward coordinate, in centimeters.
            step_cm: Cartesian step applied per UP / DOWN button
                event in xyz_control.
        """

        while True:

            self._wait_in_sleep()
            self.power_on()
            self.xyz_control(x_default, y_default, z_default, step_cm)
            self.power_off()


    # =========================================================================
    # Public lifecycle
    # =========================================================================

    def power_on(self):
        """Calibrate the encoders and move every joint into its operating range.

        Precondition: the arm is held physically at the "relaxed" pose,
        i.e. each joint at its declared ``angle_offset``. The user
        asserts this pose before invoking ``power_on``; the controller
        cannot detect violations.

        For each joint, in ``POWER_SEQUENCE`` order, the method
        (i) resets the encoder to ``angle_offset`` so that the encoder
        zero is aligned with the asserted physical pose;
        (ii) clamps ``angle_offset`` to the operating range
        ``[angle_min, angle_max]`` and drives the joint there if the
        offset itself falls outside the operating range; and
        (iii) engages an active hold via ``Joint.hold``.

        Returns:
            ``True`` if power-on completed (the arm is now powered with
            all joints holding inside their operating ranges); ``False``
            if ``power_on`` was called while ``self.powered`` was already
            True (idempotent no-op).

        Notes:
            ``self.powered`` is asserted only after the per-joint loop
            completes, so that any exception raised during calibration
            leaves the controller in the unpowered state.
        """

        if self.powered:

            print("power_on: already powered")
            return False

        print("== POWER ON ==")
        print("(precondition: arm physically at relaxed pose = angle_offset)")

        self.ev3_brick.light.on(Color.YELLOW)

        for joint_id in self.POWER_SEQUENCE:

            joint = self.joints[joint_id]

            joint.reset_angle(joint.angle_offset)
            target = _clamp(joint.angle_offset, joint.angle_min, joint.angle_max)

            if target != joint.angle_offset:

                print("M{} {}: offset {} outside [{}..{}], moving into operating range -> {}".format(joint_id, joint.name, joint.angle_offset, joint.angle_min, joint.angle_max, target))
                joint.run_to_angle(target, True, speed = POWER_SEQUENCE_SPEED_DEG_S)

            else:

                print("M{} {}: offset {} in operating range [{}..{}], holding".format(joint_id, joint.name, joint.angle_offset, joint.angle_min, joint.angle_max))

            joint.hold()

        self.powered = True

        print("== POWER ON done: all joints in operating range, holding ==")

        if self.debug:

            self.print_status()

        return True


    def power_off(self):
        """Park every joint at its ``angle_offset`` and release the motors.

        For each joint, in ``POWER_SEQUENCE`` order, the method drives
        the joint to ``angle_offset`` (the relaxed pose) with a blocking
        ``run_to_angle`` call. The operating-range clamp applied by
        ``power_on`` is intentionally NOT applied here: parking is the
        one place where it is acceptable to drive a joint to a position
        outside its normal operating range, since the relaxed pose is
        the geometry the arm is physically constructed for and the
        user has asserted that the structure can rest there safely.

        After every joint reaches its offset, every motor is set to
        coast (no holding torque), the brick LED is set to red, and
        ``self.powered`` is cleared.

        Returns:
            ``True`` if power-off completed; ``False`` if ``power_off``
            was called while ``self.powered`` was already False
            (idempotent no-op).
        """

        if not self.powered:

            print("power_off: already off")
            return False

        print("== POWER OFF ==")

        for joint_id in self.POWER_SEQUENCE:

            joint = self.joints[joint_id]
            current = joint.get_angle()
            target = joint.angle_offset

            if current != target:

                print("M{} {}: parking from {} -> relaxed pose {}".format(joint_id, joint.name, current, target))
                joint.run_to_angle(target, True, speed = POWER_SEQUENCE_SPEED_DEG_S)

            else:

                print("M{} {}: already at relaxed pose {}".format(joint_id, joint.name, target))

        for joint in self.joints:

            joint.coast()

        self.powered = False
        self.ev3_brick.light.on(Color.RED)

        print("== POWER OFF done: arm at relaxed pose, motors released ==")

        return True


    def xyz_control(self, x_default = 18, y_default = 10, z_default = 5, step_cm = 1):
        """Run the interactive Cartesian-control loop on the brick buttons and touch sensors.

        Drives the arm to an initial target ``(x_default, y_default,
        z_default)`` and then enters an event loop on the EV3 brick
        that lets the user move the target around the workspace, drive
        the gripper, and either return to the initial pose or exit
        cleanly. The loop runs at ``XYZ_TICK_PERIOD_MS`` (50 ms,
        i.e. 20 Hz, by default) and is the ACTIVE half of the
        ``run`` state machine.

        Input mapping (NORMAL sub-state):

        * LEFT / RIGHT     -- cycle the selected coordinate
          (X -> Y -> Z -> X), debounced per tap.
        * UP / DOWN        -- increment / decrement the selected
          coordinate by ``step_cm``. Held: continuous retargeting at
          the loop frequency; status printed every
          ``XYZ_PRINT_EVERY_N_TICKS`` ticks plus once on release.
        * CENTER (tap)     -- return to the initial target via the
          sequential branch of ``go_destination``.
        * CENTER (hold)    -- after ``XYZ_LONG_PRESS_TICKS`` consecutive
          ticks held (1.5 s), exit the loop cleanly.
        * Touch S1 / S4    -- close / open the gripper one
          ``GRIPPER_STEP_DEG`` per tick at ``GRIPPER_TICK_PERIOD_MS``.
        * Touch S1 + S4    -- exit the loop cleanly.
        * BACK button      -- hard kill (handled by the Pybricks runtime).

        Sub-states (mutually exclusive within a tick):

        * NORMAL          -- default; LED yellow; full input as above.
        * TOLERANCE_PAUSE -- entered when any IK joint's encoder lags
          its target by more than ``TOLERANCE_PAUSE_THRESHOLD_DEG``.
          LED orange, 600 Hz beep on entry. UP / DOWN / LEFT / RIGHT
          input is suspended; CENTER, touch sensors, and both-touch
          exit continue to work. The Pybricks PID continues pursuing
          the last commanded target via the active ``track_target``
          so the lag closes naturally. Auto-exits when every IK joint
          is back within the threshold. This is the single feedback
          mechanism layered on top of the Pybricks PID; mechanical
          stall (``is_stalled``) is left to surface here as well, via
          the resulting growth of the lag past the threshold, rather
          than via a separate state with its own recovery flow.

        Args:
            x_default: Forward coordinate of the initial target,
                in centimeters, in the base frame.
            y_default: Leftward coordinate of the initial target.
            z_default: Upward coordinate of the initial target.
            step_cm: Cartesian step applied per UP / DOWN button event.
        """

        if not self.powered:

            print("xyz_control: not powered, call power_on() first")
            return

        target = [float(x_default), float(y_default), float(z_default)]
        selected = 0
        center_held_ticks = 0
        ud_held_ticks = 0
        paused_for_tolerance = False

        print("=" * 50)
        print("XYZ CONTROL — step = {}cm, default = ({}, {}, {})".format(step_cm, x_default, y_default, z_default))
        print("  LEFT / RIGHT       = select coord (X/Y/Z, cycles)")
        print("  UP   / DOWN        = move target +/- {}cm".format(step_cm))
        print("  TOUCH right (S1)   = gripper +1° (close)")
        print("  TOUCH left  (S4)   = gripper -1° (open)")
        print("  CENTER (tap)       = return to default")
        print("  CENTER (hold 1.5s) = EXIT")
        print("  BACK button        = HARD KILL (stops program)")
        print("=" * 50)

        self.go_destination(target[0], target[1], target[2])
        self._print_xyz_status(target, selected)

        while True:

            if self.touch_sensor_right.pressed() and self.touch_sensor_left.pressed():

                print("== Both touch sensors: EXIT xyz_control ==")

                while self.touch_sensor_right.pressed() or self.touch_sensor_left.pressed():

                    wait(TOUCH_DEBOUNCE_PERIOD_MS)

                break

            joints_out_of_tol = [
                joint_id
                for joint_id in self.IK_CHAIN_JOINT_IDS
                if self.joints[joint_id].target_angle is not None
                and abs(self.joints[joint_id].get_angle() - self.joints[joint_id].target_angle) > TOLERANCE_PAUSE_THRESHOLD_DEG
            ]

            if joints_out_of_tol and not paused_for_tolerance:

                self._enter_tolerance_pause(joints_out_of_tol)
                paused_for_tolerance = True

            elif not joints_out_of_tol and paused_for_tolerance:

                self._exit_tolerance_pause()
                paused_for_tolerance = False

            if self.touch_sensor_right.pressed() or self.touch_sensor_left.pressed():

                self._handle_gripper_touch_input()
                self._print_xyz_status(target, selected)
                continue

            pressed = self.ev3_brick.buttons.pressed()

            if Button.CENTER in pressed:

                center_held_ticks += 1

                if center_held_ticks == XYZ_LONG_PRESS_TICKS:

                    print("== HOLD detected: EXIT xyz_control ==")
                    break

            else:

                if 0 < center_held_ticks < XYZ_LONG_PRESS_TICKS:

                    new_target = [float(x_default), float(y_default), float(z_default)]

                    if self.go_destination(new_target[0], new_target[1], new_target[2], sequential = True):

                        target = new_target
                        self._print_xyz_status(target, selected)

                center_held_ticks = 0

                if not paused_for_tolerance:

                    if Button.LEFT in pressed:

                        selected = (selected - 1) % 3
                        self._print_xyz_status(target, selected)
                        self._wait_button_release(Button.LEFT)

                    elif Button.RIGHT in pressed:

                        selected = (selected + 1) % 3
                        self._print_xyz_status(target, selected)
                        self._wait_button_release(Button.RIGHT)

                    elif Button.UP in pressed or Button.DOWN in pressed:

                        sign = 1 if Button.UP in pressed else -1
                        new_target = list(target)
                        new_target[selected] += sign * step_cm

                        if self.go_destination(new_target[0], new_target[1], new_target[2], tracking = True):

                            target = new_target

                        if ud_held_ticks % XYZ_PRINT_EVERY_N_TICKS == 0:

                            self._print_xyz_status(target, selected)

                        ud_held_ticks += 1

                    else:

                        if ud_held_ticks > 0:

                            self.go_destination(target[0], target[1], target[2], blocking = False)
                            self._print_xyz_status(target, selected)
                            ud_held_ticks = 0

            wait(XYZ_TICK_PERIOD_MS)


    # =========================================================================
    # Public motion API
    # =========================================================================

    def go_destination(self, x, y, z, sequential = False, blocking = True, tracking = False):
        """Drive the arm to a Cartesian target using validate-then-execute.

        Resolves ``(x, y, z)`` to joint angles via the IK solver,
        validates that every joint angle lies within its operating
        range, and only then dispatches motor commands. This guarantees
        that the arm never enters a partially-valid pose: either every
        joint moves, or none does.

        Args:
            x: Forward coordinate of the target, in centimeters,
                in the base frame.
            y: Leftward coordinate, in centimeters, in the base frame.
            z: Upward coordinate, in centimeters, in the base frame.
            sequential: If True, the joints are commanded one at a
                time in fixed kinematic order (base, shoulder, elbow),
                each blocking until it reaches its target. Used by
                the interactive ``xyz_control`` "return to default"
                action where predictability of motion matters more
                than total travel time. Implies ``blocking=True``
                semantically and ignores the ``blocking`` argument.
            blocking: If True (default), dispatch all three joints in
                parallel and then wait for them to settle, applying
                ``_correct_drift`` as a final stabilization step. If
                False, dispatch and return immediately, leaving the
                caller responsible for any subsequent waiting (used
                as the settle dispatch when the user releases UP / DOWN
                after a tracked retargeting hold).
            tracking: If True, dispatch each IK-chain joint via
                ``Joint.track_to_angle`` (Pybricks ``track_target``):
                no trapezoidal profile is planned, the position
                controller pursues the new target continuously, and
                no post-move drift correction is applied. Designed
                for the high-frequency retargeting loop in
                ``xyz_control`` (UP / DOWN held), where re-planning a
                profile every 50 ms would introduce visible jerk.
                Mutually exclusive with ``sequential``; the
                ``blocking`` argument is ignored when ``tracking`` is
                True.

        Returns:
            ``True`` if the target was accepted and dispatched;
            ``False`` if the arm is unpowered or the target was
            rejected by ``_validate_target`` (a diagnostic message
            is printed in either case).
        """

        if not self.powered:

            print("go_destination: not powered")
            return False

        targets, reason = self._validate_target(x, y, z)

        if targets is None:

            print("REJECTED ({}, {}, {}): {}".format(x, y, z, reason))
            return False

        if tracking:

            for joint_id, angle in targets.items():

                self.joints[joint_id].track_to_angle(angle)

            return True

        if sequential:

            for joint_id in self.IK_CHAIN_JOINT_IDS:

                if joint_id in targets:

                    self.joints[joint_id].run_to_angle(targets[joint_id], True)

            self._correct_drift(self.IK_CHAIN_JOINT_IDS)

            return True

        for joint_id, angle in targets.items():

            self.joints[joint_id].run_to_angle(angle, False)

        if blocking:

            self._wait_and_print(self.IK_CHAIN_JOINT_IDS)
            self._correct_drift(self.IK_CHAIN_JOINT_IDS)

        return True


    def can_reach(self, x, y, z):
        """Return True if ``(x, y, z)`` is reachable under all joint limits.

        Pure pre-check: identical validation pipeline as
        ``go_destination`` but no motor dispatch and no diagnostic
        printout. Useful as a guard in higher-level code that wants to
        plan a sequence of poses before committing to any motion.

        Args:
            x: Forward coordinate of the target, in centimeters.
            y: Leftward coordinate, in centimeters.
            z: Upward coordinate, in centimeters.

        Returns:
            ``True`` if every IK-chain joint angle for the target is
            both well-defined (target inside the reachable workspace)
            and within its operating range; ``False`` otherwise.
        """

        targets, _ = self._validate_target(x, y, z)
        return targets is not None


    # =========================================================================
    # Public utilities
    # =========================================================================

    def print_status(self, selected = None):
        """Print a tabular snapshot of every joint's encoder and target.

        For each joint, prints the joint id, kinematic name, current
        encoder angle, commanded target (or ``--`` if no target has
        been set), the residual ``current - target``, and the
        operating range. The row corresponding to ``selected`` (if
        provided) is marked with a ``>`` prefix to draw attention to
        it; this is used by the interactive ``xyz_control`` mode.

        Args:
            selected: Optional joint id to highlight in the output.
                Pass ``None`` (the default) to render an unmarked
                snapshot.
        """

        print("-" * 51)

        for joint_id, motor in enumerate(self.joints):

            marker = ">" if joint_id == selected else " "
            current = motor.get_angle()
            target = motor.target_angle
            range_str = "[{:+4d}..{:+4d}]".format(motor.angle_min, motor.angle_max)

            if target is None:

                body = "{} M{} {:6s} {:+6.1f} ->    --              ".format(marker, joint_id, motor.name, current)

            else:

                delta = target - current
                flag = "*" if motor.at_target() else " "
                body = "{} M{} {:6s} {:+6.1f} -> {:+6.1f}  d={:+6.1f} {}  ".format(marker, joint_id, motor.name, current, target, delta, flag)

            print(body + range_str)


    def coast_all(self):
        """Release every joint: cut power to all motors, allowing free motion."""

        for joint in self.joints:

            joint.coast()

        print("== coast: all motors free, manually movable ==")


    def hold_all(self):
        """Engage active hold on every joint: motors actively resist displacement."""

        for joint in self.joints:

            joint.hold()

        print("== hold: all motors locked ==")


    # =========================================================================
    # Private — IK and motion helpers
    # =========================================================================

    def _validate_target(self, x, y, z):
        """Compute the joint angles for ``(x, y, z)`` and validate every limit.

        Stages a target on the IK solver, queries each non-gripper
        joint angle, and rejects the target as soon as any joint angle
        is unreachable (target outside the reachable workspace) or
        falls outside its operating range ``[angle_min, angle_max]``.
        Implements the "validate" side of the validate-then-execute
        pattern: nothing is dispatched to any motor here.

        Args:
            x: Forward coordinate of the target, in centimeters,
                in the base frame.
            y: Leftward coordinate, in centimeters, in the base frame.
            z: Upward coordinate, in centimeters, in the base frame.

        Returns:
            A tuple ``(targets, reason)`` where exactly one element is
            non-None. On success: ``targets`` is a dict mapping each
            IK-chain joint id to its required angle in degrees, and
            ``reason`` is None. On rejection: ``targets`` is None and
            ``reason`` is a human-readable string locating the
            violation (the offending joint and the required versus
            allowed values).
        """

        self.systemXYZ.set_destination(x, y, z)
        targets = {}

        for joint_id, motor in enumerate(self.joints):

            if joint_id == 0:

                continue

            angle = self.systemXYZ.get_angle(joint_id)

            if angle is None:

                max_reach = self.systemXYZ.l1 + self.systemXYZ.l2
                min_reach = abs(self.systemXYZ.l1 - self.systemXYZ.l2)
                return None, "out of physical reach (need {:.1f} <= d <= {:.1f}cm)".format(min_reach, max_reach)

            if angle < motor.angle_min:

                return None, "M{} {} would need {:.1f}° but min={}° (under by {:.1f}°)".format(joint_id, motor.name, angle, motor.angle_min, motor.angle_min - angle)

            if angle > motor.angle_max:

                return None, "M{} {} would need {:.1f}° but max={}° (over by {:.1f}°)".format(joint_id, motor.name, angle, motor.angle_max, angle - motor.angle_max)

            targets[joint_id] = angle

        return targets, None


    def _correct_drift(self, ids, max_attempts = DRIFT_CORRECTION_MAX_ATTEMPTS):
        """Re-issue blocking moves to close any residual post-move drift.

        Backlash in the LEGO gear trains (notably the 56-tooth turntable
        on the base and the 8:40 reductions on shoulder and elbow) can
        leave a joint a few degrees off its commanded target after a
        nominally complete blocking move. This helper attempts up to
        ``max_attempts`` re-issues of ``run_to_angle`` per joint, until
        either the at-target predicate succeeds or the joint reports a
        stall. A stalled joint is logged but skipped (further re-issues
        would only saturate the controller against a mechanical limit).

        Args:
            ids: Iterable of joint ids to correct.
            max_attempts: Maximum number of re-issues per joint before
                giving up and logging a persistent drift warning.
        """

        for joint_id in ids:

            joint = self.joints[joint_id]

            if joint.target_angle is None:

                continue

            attempts = 0

            while not joint.at_target() and attempts < max_attempts:

                if joint.is_stalled():

                    delta = joint.target_angle - joint.get_angle()
                    print("M{} {}: STALLED at delta {:+.1f}°, skip correction".format(joint_id, joint.name, delta))
                    break

                delta = joint.target_angle - joint.get_angle()
                print("M{} {}: drift {:+.1f}°, correcting (attempt {}/{})".format(joint_id, joint.name, delta, attempts + 1, max_attempts))
                joint.run_to_angle(joint.target_angle, True)
                attempts += 1

            if not joint.at_target() and not joint.is_stalled():

                final_delta = joint.target_angle - joint.get_angle()
                print("M{} {}: WARNING persistent drift {:+.1f}° after {} attempts".format(joint_id, joint.name, final_delta, max_attempts))


    # =========================================================================
    # Private — interactive loop helpers
    # =========================================================================

    def _wait_in_sleep(self):
        """Block in the SLEEP state until a CENTER tap is received.

        Sets the brick LED to red (matching the powered-off
        convention), prints a single SLEEP banner, and then polls the
        button state at ``XYZ_TICK_PERIOD_MS`` until ``Button.CENTER``
        is pressed. After the press is observed, debounces by waiting
        for the release before returning, so the upstream state-machine
        transition does not see a stale pressed state.
        """

        self.ev3_brick.light.on(Color.RED)

        print("=" * 50)
        print("SLEEP — press CENTER to power on")
        print("=" * 50)

        while Button.CENTER not in self.ev3_brick.buttons.pressed():

            wait(XYZ_TICK_PERIOD_MS)

        while Button.CENTER in self.ev3_brick.buttons.pressed():

            wait(TOUCH_DEBOUNCE_PERIOD_MS)


    def _wait_and_print(self, ids, period_ms = WAIT_POLL_PERIOD_MS, timeout_ms = WAIT_TIMEOUT_MS):
        """Block until every joint in ``ids`` reports at-target or the timeout expires.

        Polls ``Joint.at_target`` at ``period_ms`` intervals; if every
        named joint is within tolerance, returns immediately. While
        polling, optionally prints a per-tick status snapshot
        (``self.print_status()``) when ``self.debug`` is True. After
        the loop exits (either all-at-target or timeout), prints a
        terminal snapshot followed by a ``== done ==`` marker.

        Args:
            ids: Iterable of joint ids to monitor.
            period_ms: Polling period, in milliseconds.
            timeout_ms: Upper bound on the total polling time, in
                milliseconds. The function returns after this duration
                even if some joints have not yet reached their targets.
        """

        elapsed = 0

        while elapsed < timeout_ms:

            if all(self.joints[joint_id].at_target() for joint_id in ids):

                break

            if self.debug:

                self.print_status()
                print("--")

            wait(period_ms)
            elapsed += period_ms

        if self.debug:

            self.print_status()
            print("== done ==")


    def _wait_button_release(self, button):
        """Block until ``button`` is no longer being pressed on the brick.

        Polls the brick button state at ``TOUCH_DEBOUNCE_PERIOD_MS``
        intervals; intended to debounce a single press into a single
        action by the caller (LEFT and RIGHT in ``xyz_control``).

        Args:
            button: A ``pybricks.parameters.Button`` constant
                identifying the button to wait on.
        """

        while button in self.ev3_brick.buttons.pressed():

            wait(TOUCH_DEBOUNCE_PERIOD_MS)


    def _print_xyz_status(self, target, selected):
        """Print the current Cartesian target plus a joint-level snapshot.

        Renders the three components of ``target`` on a single line,
        marking the ``selected`` axis with a ``>`` prefix, then defers
        to ``print_status`` for the per-joint encoder readout. Used by
        ``xyz_control`` whenever the target or selected axis changes.

        Args:
            target: Three-element sequence ``(x, y, z)`` of Cartesian
                coordinates, in centimeters, in the base frame.
            selected: Index in ``{0, 1, 2}`` of the currently selected
                axis (0 = X, 1 = Y, 2 = Z).
        """

        coord_names = ["X", "Y", "Z"]
        line = " TARGET: "

        for coord_index in range(3):

            marker = ">" if coord_index == selected else " "
            line += "{}{}={:+6.1f}cm  ".format(marker, coord_names[coord_index], target[coord_index])

        print("-" * 51)
        print(line)
        self.print_status()


    def _handle_gripper_touch_input(self):
        """Drive the gripper from touch-sensor input while at least one is pressed.

        Loops at ``GRIPPER_TICK_PERIOD_MS`` while either touch sensor is
        held down, incrementing the gripper angle by ``GRIPPER_STEP_DEG``
        on the right sensor (close) or decrementing on the left sensor
        (open), clamped to the gripper's operating range. Returns
        immediately if both sensors become pressed simultaneously,
        leaving the outer ``xyz_control`` loop to recognise the
        both-pressed signal and exit on its next iteration.
        """

        gripper = self.joints[0]
        angle = gripper.get_angle()
        tick_counter = 0

        while self.touch_sensor_right.pressed() or self.touch_sensor_left.pressed():

            if self.touch_sensor_right.pressed() and self.touch_sensor_left.pressed():

                break

            if self.touch_sensor_right.pressed():

                angle += GRIPPER_STEP_DEG

            else:

                angle -= GRIPPER_STEP_DEG

            angle = _clamp(angle, gripper.angle_min, gripper.angle_max)
            gripper.run_to_angle(angle, False)

            if self.debug and tick_counter % GRIPPER_DEBUG_PRINT_EVERY_N == 0:

                self.print_status()

            tick_counter += 1
            wait(GRIPPER_TICK_PERIOD_MS)


    def _enter_tolerance_pause(self, joint_ids):
        """Enter the tolerance-pause sub-state: freeze coordinate input until the controller catches up.

        Triggered when ``xyz_control`` detects that any IK joint's
        encoder lags its commanded target by more than
        ``TOLERANCE_PAUSE_THRESHOLD_DEG``. The brick LED switches to
        orange and the speaker emits a 600 Hz advisory beep. UP, DOWN,
        LEFT, and RIGHT input is suspended in the xyz_control loop
        while this sub-state is active, but CENTER tap (return to
        default), CENTER hold (exit), touch-sensor gripper input, and
        both-touch exit all continue to function. The Pybricks PID
        controller continues pursuing the last commanded target via
        the active ``track_target`` so the lag closes naturally; no
        additional re-issue is performed at this layer.

        Args:
            joint_ids: Iterable of the joint ids that triggered the
                pause, used only for the diagnostic printout. The
                pause is whole-arm regardless of how many joints
                exceeded the threshold.
        """

        self.ev3_brick.light.on(Color.ORANGE)
        self.ev3_brick.speaker.beep(frequency = 600, duration = 100)

        print("== TOLERANCE PAUSE: input frozen until joint(s) recover ==")

        for joint_id in joint_ids:

            joint = self.joints[joint_id]
            delta = joint.target_angle - joint.get_angle()
            print("  M{} {}: |delta|={:.1f}° (threshold {}°)".format(joint_id, joint.name, abs(delta), TOLERANCE_PAUSE_THRESHOLD_DEG))


    def _exit_tolerance_pause(self):
        """Exit the tolerance-pause sub-state after the controller closes the lag.

        Restores the powered-on yellow LED and prints a recovery
        acknowledgement. Called from ``xyz_control`` once every IK
        joint's lag has fallen back below
        ``TOLERANCE_PAUSE_THRESHOLD_DEG``.
        """

        self.ev3_brick.light.on(Color.YELLOW)
        print("== Recovered: tolerance restored, resuming input ==")


class Joint:
    """Per-joint controller built on top of the Pybricks ev3devices.Motor API.

    This class is the only place in the module that touches a Pybricks
    motor object directly. It augments the Pybricks motor with:

    - per-joint operating limits (``angle_min``, ``angle_max``);
    - a kinematic offset (``angle_offset``) used at power-on calibration
      to align the encoder zero with a known physical pose;
    - a cached ``target_angle`` carrying the most recent commanded
      target, so that drift and at-target checks (``at_target``,
      ``_correct_drift`` in the parent class) can reason about the
      joint without re-querying it;
    - a uniform position tolerance (``POSITION_TOLERANCE_DEG``) shared
      with the Pybricks internal position controller via
      ``control.target_tolerances``;
    - a uniform speed and acceleration envelope (``MOTOR_MAX_SPEED``,
      ``MOTOR_ACCELERATION``) configured at construction via
      ``control.limits``, so that every motion primitive
      (``run_target``, ``track_target``, ``run``, etc.) shares the
      same hard caps on the underlying PID controller.

    The ``device`` attribute holds the underlying ``pybricks.ev3devices.
    Motor`` instance and is intentionally kept private in spirit:
    consumers of ``ThreeAxisRobotArm`` should never reach through it.
    """

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(self, device, angle_speed, angle_offset, angle_max, angle_min, name = ""):
        """Wrap a Pybricks motor with kinematic limits and a target cache.

        Args:
            device: A ``pybricks.ev3devices.Motor`` instance, already
                configured with port, direction and external gear train.
            angle_speed: Default angular speed for ``run_to_angle``,
                in degrees per second on the output shaft.
            angle_offset: Encoder value, in degrees, that corresponds
                to the physical pose held by the joint at power-on
                calibration time. Used by ``ThreeAxisRobotArm.power_on``
                to call ``reset_angle(angle_offset)``.
            angle_max: Upper operating limit of the joint, in degrees,
                measured in the same convention as the encoder readings.
            angle_min: Lower operating limit of the joint, in degrees.
            name: Short human-readable label used in debug printouts
                (``"base"``, ``"shldr"``, ``"elbow"``, ``"grip"``).

        Notes:
            The Pybricks position controller is configured here with
            a position tolerance equal to ``POSITION_TOLERANCE_DEG``,
            so that the controller's own at-target predicate matches
            the one used by ``at_target`` and the drift correction in
            the parent class. The PID hard caps on speed and
            acceleration are set via ``control.limits`` to
            ``MOTOR_MAX_SPEED`` and ``MOTOR_ACCELERATION``
            respectively, so that ``run_target`` and ``track_target``
            share a single, predictable motion envelope.
        """

        self.device = device
        self.angle_speed = angle_speed
        self.angle_offset = angle_offset
        self.angle_max = angle_max
        self.angle_min = angle_min
        self.name = name
        self.target_angle = None

        self.device.control.target_tolerances(speed = 5, position = POSITION_TOLERANCE_DEG)
        self.device.control.limits(speed = MOTOR_MAX_SPEED, acceleration = MOTOR_ACCELERATION)


    # =========================================================================
    # State queries
    # =========================================================================

    def get_angle(self):
        """Return the current encoder angle, in degrees on the output shaft."""

        return self.device.angle()


    def at_target(self, tolerance = POSITION_TOLERANCE_DEG):
        """Return True if the joint is within ``tolerance`` degrees of its cached target.

        Args:
            tolerance: Allowed absolute deviation, in degrees. Defaults
                to ``POSITION_TOLERANCE_DEG`` (the same value used by
                the Pybricks internal controller and by the drift
                correction), so that all three layers agree on the
                at-target predicate.

        Returns:
            ``True`` if no target has been commanded yet (the cached
            ``target_angle`` is ``None``) or if the current encoder
            angle lies within ``tolerance`` of the cached target;
            ``False`` otherwise.
        """

        if self.target_angle is None:

            return True

        return abs(self.get_angle() - self.target_angle) <= tolerance


    def is_stalled(self):
        """Return True if the underlying Pybricks controller reports a stall."""

        return self.device.control.stalled()


    # =========================================================================
    # Calibration
    # =========================================================================

    def reset_angle(self, angle):
        """Reset the encoder so that the current physical pose reads ``angle``.

        Used at power-on calibration: the caller positions the joint
        physically at a known pose, then asserts that pose by passing
        the corresponding kinematic angle here. The cached
        ``target_angle`` is cleared so that ``at_target`` does not
        report success against a stale target.

        Args:
            angle: Encoder value, in degrees, to assign to the current
                physical position.
        """

        self.device.reset_angle(angle)
        self.target_angle = None


    # =========================================================================
    # Motion dispatch
    # =========================================================================

    def run_to_angle(self, angle, blocking = True, speed = None):
        """Drive the joint to an absolute target angle along a motion profile.

        Wraps ``pybricks.ev3devices.Motor.run_target`` with the
        joint's default ``angle_speed`` (or an explicit per-call
        override) and caches the commanded target so that downstream
        tolerance and drift logic can reason about it.

        Args:
            angle: Absolute target angle on the output shaft,
                in degrees.
            blocking: If True (default), block until the controller
                reports the target reached (within
                ``POSITION_TOLERANCE_DEG``). If False, dispatch the
                command and return immediately, allowing parallel
                multi-joint moves.
            speed: Optional override for the cruise speed of the
                trapezoidal profile, in deg / s on the output shaft.
                Defaults to ``self.angle_speed`` (set at construction).
                Used by the power-on / power-off sweeps to dispatch
                slower, gentler parking moves than the operating-mode
                default. Always clamped to ``MOTOR_MAX_SPEED`` by the
                Pybricks controller via ``control.limits(speed)``.
        """

        self.target_angle = angle
        self.device.run_target(speed if speed is not None else self.angle_speed, angle, wait = blocking)


    def track_to_angle(self, angle):
        """Track a (possibly time-varying) absolute target without a motion profile.

        Wraps ``pybricks.ev3devices.Motor.track_target``: the position
        controller pursues the supplied target continuously, without
        re-planning a trapezoidal profile per call. Useful for high-
        frequency retargeting scenarios where ``run_to_angle`` would
        otherwise schedule and discard a fresh profile per tick.

        Args:
            angle: Absolute target angle on the output shaft, in degrees.
        """

        self.target_angle = angle
        self.device.track_target(angle)


    # =========================================================================
    # Stop modes
    # =========================================================================

    def hold(self):
        """Actively hold the current position (powered, resists displacement)."""

        self.device.hold()


    def coast(self):
        """Release the joint: cut power and let it move freely.

        The underlying ``device.stop()`` is the Pybricks v2 spelling
        for "stop and coast" (no holding torque), not for an active
        brake. ``device.brake()`` would apply passive braking, and
        ``device.hold()`` would actively hold position.
        """

        self.device.stop()


class SystemXYZ:
    """Analytical inverse kinematics solver for a 3-axis RRR manipulator.

    Computes the joint angles (theta_1, theta_2, theta_3) required to position
    the gripper grip point at a Cartesian target (x, y, z), using a closed-form
    decomposition into a base-yaw rotation followed by a planar two-link
    geometry solved by the law of cosines.

    Coordinate frames
    -----------------
    The user-facing input frame is centered at the joint 1 (base) axis:
        +x : forward (the neutral pointing direction with theta_1 = 0)
        +y : leftward, viewed from behind the robot
        +z : upward, opposite to gravity
    The frame is right-handed.

    The joint 2 (shoulder) axis sits a distance ``shoulder_z_offset`` above
    the base on +z. Because the planar IK formulas are most concise when
    expressed in the shoulder-centered frame, ``set_destination`` translates
    the user target by ``-shoulder_z_offset`` along z and stores the result
    internally; ``get_angle`` then operates directly on the translated target.

    Joint angle conventions
    -----------------------
    theta_1 (base)
        atan2(y, x). Zero is aligned with +x; positive values denote
        counter-clockwise rotation viewed from above.
    theta_2 (shoulder)
        alpha + beta, where alpha = atan2(z, r) is the elevation of the
        target line above the horizontal and beta is the interior angle
        of the planar triangle at the shoulder. Zero corresponds to the
        upper arm horizontal and pointing forward; positive values
        elevate the arm.
    theta_3 (elbow)
        180 - gamma, where gamma is the interior angle of the planar
        triangle at the elbow. Zero corresponds to the forearm fully
        extended (collinear with the upper arm); positive values flex
        the elbow such that the grip point approaches the base.

    The "elbow-up" branch is selected by the positive sign on beta in
    theta_2; the alternative "elbow-down" branch is unreachable on this
    manipulator due to the joint 2 lower limit.

    References
    ----------
    [1] Lynch, K. M. and Park, F. C. (2017). Modern Robotics: Mechanics,
        Planning, and Control. Cambridge University Press, sec. 6.1.
    [2] Spong, M. W., Hutchinson, S., and Vidyasagar, M. (2005). Robot
        Modeling and Control. Wiley, ch. 3.
    """

    def __init__(self, l1, l2, shoulder_z_offset = 0):
        """Initialize the solver with the manipulator geometry.

        Args:
            l1: Upper-arm link length, in centimeters (between the
                shoulder axis and the elbow axis).
            l2: Forearm link length, in centimeters (between the elbow
                axis and the gripper grip point).
            shoulder_z_offset: Vertical distance from the base axis
                (joint 1) to the shoulder axis (joint 2), in centimeters.
                Positive values place the shoulder above the base, which
                is the typical configuration. Defaults to 0, which makes
                the two axes coplanar and reduces the model to a planar
                two-link arm mounted on a vertical revolute base.
        """

        self.l1 = l1
        self.l2 = l2
        self.shoulder_z_offset = shoulder_z_offset

        # Pre-squared link lengths, reused by the law-of-cosines invocations below.
        self._l1_squared = l1 * l1
        self._l2_squared = l2 * l2

        # Cached state filled by set_destination; reused across the three
        # get_angle calls per IK query (one per joint) to avoid recomputing
        # the planar radius r and the 3D distance d three times.
        self._target_shoulder_frame = (0.0, 0.0, 0.0)
        self._horizontal_radius = 0.0
        self._target_distance = 0.0
        self._is_reachable = False


    def set_destination(self, x, y, z):
        """Set the Cartesian target for the gripper grip point.

        The grip point is the geometric location at which the gripper
        jaws meet when fully closed, i.e. the application point of the
        prehension force. It lies at the end of the forearm, at distance
        ``l2`` from the elbow axis along the forearm direction.

        Args:
            x: Forward coordinate, in centimeters, in the base frame.
            y: Leftward coordinate, in centimeters, in the base frame.
            z: Upward coordinate, in centimeters, in the base frame.
                ``z = 0`` corresponds to the level of the base axis.

        Notes:
            The target is stored after translation into the shoulder
            frame (the z component is shifted by ``-shoulder_z_offset``),
            so that subsequent calls to ``get_angle`` apply the planar
            IK formulas directly without a per-call frame conversion.

            The horizontal radius ``r = sqrt(x^2 + y^2)``, the 3D distance
            ``d = sqrt(r^2 + z_shoulder^2)``, and the boolean reachability
            predicate (``|l_1 - l_2| <= d <= l_1 + l_2``) are also cached
            here, since each is invariant across the three joints and
            would otherwise be recomputed three times per IK query.
        """

        z_shoulder = z - self.shoulder_z_offset
        self._target_shoulder_frame = (x, y, z_shoulder)

        r = sqrt(x * x + y * y)
        d = sqrt(r * r + z_shoulder * z_shoulder)
        self._horizontal_radius = r
        self._target_distance = d

        self._is_reachable = abs(self.l1 - self.l2) <= d <= self.l1 + self.l2


    def get_angle(self, joint_id):
        """Compute the joint angle required to reach the current target.

        Implements the analytical inverse kinematics for the 3R RRR
        chain: the base yaw is decoupled by atan2 and the resulting
        planar two-link sub-problem in the vertical plane is solved
        in closed form via the law of cosines (elbow-up branch).

        Args:
            joint_id: Index of the IK chain joint. Must be 1 (base),
                2 (shoulder), or 3 (elbow). The gripper joint (id 0)
                is not part of the IK chain and is rejected.

        Returns:
            The joint angle in degrees, or ``None`` if the current
            target (set via the most recent ``set_destination`` call)
            lies outside the reachable workspace, i.e. the cached 3D
            distance does not satisfy
            ``|l_1 - l_2| <= d <= l_1 + l_2``.

        Raises:
            ValueError: If ``joint_id`` is not in {1, 2, 3}.

        Notes:
            With ``r = sqrt(x^2 + y^2)`` the horizontal projection of
            the target and ``d = sqrt(r^2 + z^2)`` its 3D distance from
            the shoulder, the closed-form expressions are:

                theta_1 = atan2(y, x)
                theta_2 = alpha + beta
                    alpha = atan2(z, r)
                    beta  = acos((l_1^2 + d^2 - l_2^2) / (2 l_1 d))
                theta_3 = 180 deg - gamma
                    gamma = acos((l_1^2 + l_2^2 - d^2) / (2 l_1 l_2))

            Both ``alpha`` and the two law-of-cosines arccosines reduce
            to well-defined values in [0, 180 deg] under the cached
            reachability predicate, so no additional clamping is
            required at this layer.

        References:
            Lynch, K. M. and Park, F. C. (2017). Modern Robotics, sec. 6.1.
        """

        if joint_id not in (1, 2, 3):

            raise ValueError(
                "joint_id must be 1 (base), 2 (shoulder), or 3 (elbow); got {}".format(joint_id)
            )

        if not self._is_reachable:

            return None

        x, y, z = self._target_shoulder_frame
        r = self._horizontal_radius
        d = self._target_distance

        if joint_id == 1:

            return degrees(atan2(y, x))

        if joint_id == 2:

            alpha = degrees(atan2(z, r))
            beta = degrees(acos((self._l1_squared + d * d - self._l2_squared) / (2 * self.l1 * d)))

            return alpha + beta

        # joint_id == 3 (elbow); the validation at the top of the function
        # has already rejected any other value with a ValueError.
        gamma = degrees(acos((self._l1_squared + self._l2_squared - d * d) / (2 * self.l1 * self.l2)))

        return 180 - gamma
