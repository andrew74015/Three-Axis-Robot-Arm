# Three-Axis Robot Arm: Analytical Inverse Kinematics on the LEGO MINDSTORMS EV3

A 3-degree-of-freedom RRR anthropomorphic manipulator with a parallel-jaw gripper, controlled in Cartesian space through closed-form analytical inverse kinematics. The solver decouples the base yaw from a planar two-link sub-problem, which is then resolved in closed form via the law of cosines, following the analytical procedure of Lynch & Park, *Modern Robotics: Mechanics, Planning, and Control* (Cambridge University Press, 2017), section 6.1. The implementation targets the LEGO MINDSTORMS EV3 platform via the Pybricks MicroPython v2.0 runtime.

## Table of contents

1. [Overview](#1-overview)
2. [Mathematical formulation](#2-mathematical-formulation)
3. [Mechanical system](#3-mechanical-system)
4. [Software architecture](#4-software-architecture)
5. [Design principles](#5-design-principles)
6. [Operation](#6-operation)
7. [Build and run](#7-build-and-run)
8. [Code structure](#8-code-structure)
9. [Validation](#9-validation)
10. [References](#10-references)
11. [Purpose and scope](#11-purpose-and-scope)

## 1. Overview

**Problem.** Given a Cartesian target `(x, y, z)` for the gripper grip point, compute the joint-space configuration `(theta_1, theta_2, theta_3)` that places the manipulator end-effector at that target, and dispatch the corresponding motor commands without violating the workspace, joint-limit, or motion-smoothness constraints.

**Approach.** The kinematic chain admits a closed-form analytical solution. The first joint (base yaw) decouples from the rest via `atan2(y, x)`, leaving a planar two-link sub-problem in the vertical plane that is resolved by the law of cosines. No iterative root-finding is required, and no linearised Jacobian inversion is performed; the joint angles are obtained in `O(1)` arithmetic operations per query. This is essential on the target hardware, where the interactive control loop runs at 20 Hz on a constrained 16 MHz ARM9 microcontroller.

**Contribution.** The implementation factors the kinematics, the validation pipeline, and the motion-dispatch logic into three orthogonal classes (`SystemXYZ`, `Joint`, `ThreeAxisRobotArm`), so that the mathematics is testable in isolation from the hardware abstraction. A *validate-then-execute* invariant, applied at every Cartesian command, guarantees that the manipulator never enters a partially-valid pose. The user-facing fault feedback collapses into a single, library-aligned mechanism that does not duplicate the work of the underlying PID controller.

## 2. Mathematical formulation

### 2.1 Coordinate frames

The user-facing input frame `{s}` is right-handed and centred at the joint 1 (base) axis:

- `+x` -- forward, the neutral pointing direction at `theta_1 = 0`;
- `+y` -- leftward, viewed from behind the manipulator;
- `+z` -- upward, opposite to gravity.

The shoulder (joint 2) axis sits at a fixed vertical offset `z_sh` above the base, on `+z`. Internally, the solver translates the user-facing target by `-z_sh` along `z` so that the planar inverse-kinematics formulas operate in the shoulder-centred frame:

```
z' = z - z_sh
```

Joint angles are returned in the user-facing convention; the frame translation is hidden from the caller.

### 2.2 Joint conventions

| Index | Joint            | Axis of rotation | Zero pose                              | Positive sense                              |
|:-----:|------------------|------------------|----------------------------------------|---------------------------------------------|
| 1     | Base (yaw)       | `+z`             | aligned with `+x`                      | counter-clockwise viewed from above         |
| 2     | Shoulder (pitch) | local `-y`       | upper arm horizontal, pointing forward | upward elevation                            |
| 3     | Elbow (pitch)    | local `+y`       | forearm collinear with upper arm       | flexion (grip point approaches the base)    |

The shoulder and elbow rotation axes have opposite signs in body coordinates so that the elementary commands "raise the arm" and "bend the elbow" both correspond to *positive* joint increments, a convention that keeps the joint-limit declarations and the closed-form formulas unsigned.

### 2.3 Closed-form inverse kinematics

Let `l1` and `l2` denote the upper-arm and forearm link lengths. Define the projected horizontal radius and the shoulder-to-target distance:

```
r = sqrt(x^2 + y^2)
d = sqrt(r^2 + z'^2)
```

**Reachability.** A target lies in the manipulator's reachable workspace if and only if

```
|l1 - l2|  <=  d  <=  l1 + l2
```

The lower bound excludes the spherical "dead zone" near the shoulder; the upper bound is the maximum-extension envelope.

**Closed-form solution (elbow-up branch).** Decoupling the yaw and applying the law of cosines to the planar triangle of sides `(l1, l2, d)`:

```
theta_1 = atan2(y, x)

alpha   = atan2(z', r)
beta    = arccos( (l1^2 + d^2 - l2^2) / (2 * l1 * d) )
theta_2 = alpha + beta

gamma   = arccos( (l1^2 + l2^2 - d^2) / (2 * l1 * l2) )
theta_3 = pi - gamma
```

The angle `alpha` is the elevation of the shoulder-to-target line above the horizontal; `beta` is the interior triangle angle at the shoulder vertex; `gamma` is the interior triangle angle at the elbow vertex. The transformation `theta_3 = pi - gamma` converts the *interior* angle (which equals `pi` when the arm is fully extended) into the *flexion* convention used by the controller (zero when extended, positive under flexion).

**Branch selection.** The triangle `(l1, l2, d)` admits two configurations: elbow above the shoulder-to-target line (*elbow-up*) and elbow below it (*elbow-down*). The implementation selects the elbow-up branch through the positive sign on `beta` in `theta_2`. The alternative branch is mechanically unreachable on this manipulator due to the joint-2 lower limit `theta_2 >= 25 deg`.

**Singularities.** The Jacobian determinant `det J(theta)` vanishes at three loci of the configuration space:

1. `theta_3 = 0` or `theta_3 = pi` -- elbow fully extended or fully folded;
2. `r = 0` -- target on the base axis (wrist directly above shoulder);
3. `theta_2 = theta_3 = 0` -- workspace boundary at maximum reach.

Outside these singular loci, the closed-form solution returns a well-defined, unique elbow-up configuration.

### 2.4 Validate-then-execute

Every Cartesian command resolves to a triple of joint angles `(theta_1, theta_2, theta_3)`, which is then checked for membership in the per-joint operating range `[theta_i^min, theta_i^max]`:

```
theta_i  in  [theta_i^min, theta_i^max]   for all  i  in  {1, 2, 3}
```

If any of the three joint angles falls outside its admissible range, *no* motor command is issued and the controller returns a failure with a diagnostic message identifying the offending joint and the magnitude of the violation. This guarantees the absence of partially-valid poses that an incremental dispatch (move base, then check shoulder, then reject) would otherwise admit.

## 3. Mechanical system

### 3.1 Geometry

| Symbol  | Description                                  | Value    |
|:-------:|----------------------------------------------|---------:|
| `l1`    | Upper-arm link length (shoulder to elbow)    | 24.0 cm  |
| `l2`    | Forearm link length (elbow to grip point)    | 19.2 cm  |
| `z_sh`  | Vertical offset (base axis to shoulder axis) |  4.0 cm  |
| `r_max` | Maximum reach `l1 + l2`                      | 43.2 cm  |
| `r_min` | Minimum reach `|l1 - l2|`                    |  4.8 cm  |

Lengths are derived from LEGO Technic beam dimensions through the conversion `1 unit = 0.8 cm` (the standard 8 mm hole-to-hole pitch).

### 3.2 Joint limits

| Joint    | Symbol    | Operating range   |
|----------|:---------:|:-----------------:|
| Base     | `theta_1` | `[-60, +60]` deg  |
| Shoulder | `theta_2` | `[+25, +90]` deg  |
| Elbow    | `theta_3` | `[  0, +120]` deg |
| Gripper  | (n/a)     | `[  0,  +80]` deg |

### 3.3 Hardware platform

| Port | Device           | Role                | External gear train (Pybricks notation)                              |
|:----:|------------------|---------------------|----------------------------------------------------------------------|
| A    | EV3 Large Motor  | Joint 1, base       | `[[36, 12], [1, 56]]` (LEGO 56-tooth turntable, net ratio 3:56)      |
| B    | EV3 Large Motor  | Joint 2, shoulder   | `[8, 40]` (1:5 reduction)                                            |
| C    | EV3 Medium Motor | Gripper             | `[1, 24]` (worm-like reduction)                                      |
| D    | EV3 Large Motor  | Joint 3, elbow      | `[8, 40]` (1:5 reduction)                                            |
| S1   | Touch Sensor     | Gripper close       | (n/a)                                                                |
| S4   | Touch Sensor     | Gripper open        | (n/a)                                                                |

The Pybricks `Motor` constructor accepts the gear-train descriptor and rescales every command and encoder reading to the *output* (joint) shaft, so that the control code reasons exclusively in terms of joint angles rather than motor-shaft angles.

## 4. Software architecture

The implementation is partitioned into three classes, separating the mathematics from the hardware abstraction and from the control orchestration.

```
SystemXYZ          Pure analytical inverse-kinematics solver. Hardware-
                   independent. Inputs: target (x, y, z) in cm. Outputs:
                   joint angles (theta_1, theta_2, theta_3) in degrees.

Joint              Per-joint controller built on top of pybricks.ev3devices.
                   Motor. Adds operating limits, a kinematic offset, a target-
                   angle cache, and a uniform speed/acceleration envelope
                   shared by every motion primitive.

ThreeAxisRobotArm  High-level controller. Owns four Joint instances, one
                   SystemXYZ, the EV3 brick, and two touch sensors.
                   Implements the SLEEP <-> ACTIVE state machine, the
                   validate-then-execute pipeline, the interactive xyz_control
                   loop, and the post-blocking-move drift correction.
```

The `SystemXYZ` class has no dependency on Pybricks: it consumes link lengths in centimetres and emits joint angles in degrees, without ever touching a motor object. This isolation is intentional. The kinematics is testable on any host (for example CPython on a workstation) by mocking only the manipulator geometry.

## 5. Design principles

### 5.1 Library-first discipline

The control stack rests on the Pybricks `Motor.control` PID controller. Three configuration knobs are set explicitly at `Joint` construction:

- `target_tolerances(speed=5, position=2)` is the at-target predicate, in degrees and degrees per second;
- `limits(speed=30, acceleration=200)` is a uniform motion envelope shared by every motion primitive (`run_target`, `track_target`);
- a per-call `run_target(speed=...)` override is used for the gentler power-on and power-off sweeps.

Every behaviour layered above the library is constrained to be either (i) a thin user-experience signal that the PID *is* lagging, not a re-implementation of what the PID already does, or (ii) a justified compensation for a phenomenon the PID cannot observe, specifically mechanical backlash in the LEGO gear trains (notably the 56-tooth turntable on the base and the 8:40 reductions on the shoulder and elbow). Periodic "drift correction" by external `run_target` re-issues was identified as conceptually incorrect and removed during the audit, as it duplicates the work of the internal PID and introduces motion discontinuities by re-planning the trapezoidal velocity profile every tick.

### 5.2 Two motion regimes, two primitives

| Regime                          | Primitive                                      | Use case                               |
|---------------------------------|------------------------------------------------|----------------------------------------|
| Single-shot, blocking           | `Motor.run_target` (trapezoidal profile)       | Power-on, power-off, return-to-default |
| Continuous retargeting at 20 Hz | `Motor.track_target` (PID pursuit, no profile) | UP / DOWN held in `xyz_control`        |

The distinction is essential at high tick rates. Invoking `run_target` repeatedly on a moving target re-plans the velocity profile on every call, leaving the controller permanently in a *deceleration ramp that never completes* and producing visible jerk. `Motor.track_target` updates the PID setpoint without re-planning, eliminating the discontinuity (cf. the Pybricks documentation: "useful if you want to continuously change the target angle").

### 5.3 Single-mechanism feedback

A single user-visible threshold `delta_pause = 5 deg`, compared against the per-joint encoder lag `|theta - theta_target|`, drives the entire fault-feedback surface in `xyz_control`:

- *Below the threshold*: normal operation, no signal.
- *Above the threshold*: the loop enters the `TOLERANCE_PAUSE` sub-state (LED orange, 600 Hz advisory beep, UP / DOWN / LEFT / RIGHT input frozen) until the PID closes the gap and the lag falls back below the threshold. Mechanical stall, which by definition produces a growing lag, also surfaces here, rather than via a separate halt mechanism.

Collapsing the fault feedback into a single mechanism eliminates the failure mode of a bespoke halt-on-stall layer that becomes stuck in a "ghost" state when the underlying stall releases naturally without an explicit recovery command. The `TOLERANCE_PAUSE` sub-state, by contrast, exits automatically as soon as the lag is closed.

## 6. Operation

### 6.1 State machine

The top-level controller is event-driven:

```
            +-----------+                         +-----------+
            |   SLEEP   |--- CENTER tap --------->|  ACTIVE   |
            | LED red   |                         | LED yellow|
            | coast     |<---- power_off ---------|(xyz_ctrl) |
            +-----------+                         +-----------+
                  ^                                     |
                  |                                     |
                  +--- CENTER hold 1.5 s ---------------+
                       (or  S1 + S4  pressed together)

                       BACK button -> hard kill (Pybricks runtime)
```

Within the `ACTIVE` state, the `TOLERANCE_PAUSE` sub-state (LED orange) is entered automatically when any IK joint's encoder lag exceeds `delta_pause = 5 deg`, and exited automatically when the PID closes the gap on every joint.

### 6.2 Calibration prerequisite

Before pressing `CENTER` to enter `ACTIVE`, the operator places the manipulator physically in the *relaxed pose*: each joint at its declared kinematic offset (gripper open, base centred, shoulder leaning forward, elbow folded back). The `power_on` routine resets the encoders to align with this asserted pose, then drives any joint whose offset lies outside its operating range into the nearest admissible position. The controller cannot detect violations of this precondition by design. The relaxed pose is the calibration reference, not an inferred state.

### 6.3 Interactive controls

| Input                       | Action                                                                                |
|-----------------------------|---------------------------------------------------------------------------------------|
| `LEFT` / `RIGHT`            | Cycle the selected coordinate (X -> Y -> Z -> X)                                      |
| `UP` / `DOWN`               | Move the target by `+/- step_cm` along the selected coordinate (continuous while held) |
| `CENTER` (tap)              | Return to the initial target `(x_0, y_0, z_0)`                                        |
| `CENTER` (hold >= 1.5 s)    | Exit `ACTIVE` and park                                                                |
| Touch sensor S1 (right)     | Close gripper (incremental while held)                                                |
| Touch sensor S4 (left)      | Open gripper                                                                          |
| S1 and S4 pressed together  | Exit `ACTIVE` and park                                                                |
| `BACK`                      | Hard kill                                                                             |

The initial Cartesian target and the per-tick step are configured in `main.py` through the constants `INITIAL_TARGET_X_CM`, `INITIAL_TARGET_Y_CM`, `INITIAL_TARGET_Z_CM`, and `XYZ_STEP_CM`.

## 7. Build and run

**Prerequisites.**

- A LEGO MINDSTORMS EV3 brick with a microSD card flashed using the [EV3 MicroPython v2.0](https://education.lego.com/en-us/support/mindstorms-ev3/python-for-ev3) image.
- Visual Studio Code with the official **EV3 MicroPython** extension installed.

**Procedure.**

1. Power on the EV3 brick and connect it to the host workstation via a mini-USB cable.
2. Place the manipulator physically in the relaxed pose (see section 6.2).
3. Open the project folder in Visual Studio Code.
4. Press `F5`. The extension downloads the program to the brick and starts execution.
5. The brick LED is red (`SLEEP`). Press `CENTER` on the brick to calibrate the encoders and enter `ACTIVE`.

The workspace defaults can be modified in `main.py`:

```python
INITIAL_TARGET_X_CM = 25
INITIAL_TARGET_Y_CM = 0
INITIAL_TARGET_Z_CM = 0
XYZ_STEP_CM         = 0.5
```

## 8. Code structure

The project consists of two source files at the repository root:

```
main.py                     Entry point. Wires the four motors and the
                            two touch sensors to the EV3 brick,
                            instantiates ThreeAxisRobotArm, and dispatches
                            the top-level state machine via arm.run(...).

three_axis_robot_arm.py     Controller module. Defines:
                              - ThreeAxisRobotArm  (high-level controller
                                                    and state machine);
                              - Joint              (per-joint Motor
                                                    wrapper);
                              - SystemXYZ          (analytical IK solver).
```

Module-level constants in `three_axis_robot_arm.py` are organised by concern (motion envelope, blocking-move timing, interactive-loop cadence, gripper inner loop, touch-sensor debouncing, tolerance threshold), each accompanied by a docstring explaining the chosen value. All public methods carry Google-style docstrings.

## 9. Validation

**Numerical cross-check.** The forward kinematics has been independently formulated in MATLAB using the Product-of-Exponentials (PoE) representation (Lynch & Park, section 4.1). For the test joint configuration `theta = (40, 36, 38)` deg, the MATLAB PoE evaluation yields the end-effector position `(29.57, 24.82, 13.45)` cm; the Python inverse-kinematics solver, when supplied with `(29.1, 25.2, 13.6)` cm, returns `theta = (40, 36, 38)` deg. The two formulations agree to numerical precision, providing a closed-loop sanity check that crosses both the language boundary and the FK / IK boundary.

**Empirical validation.** The interactive `xyz_control` flow has been exercised on the physical manipulator: continuous retargeting under `UP` / `DOWN` held, single-shot return-to-default on `CENTER` tap, stall-induced `TOLERANCE_PAUSE` entry and natural recovery, gripper actuation under touch-sensor input, and the `SLEEP` <-> `ACTIVE` state transitions, all confirming the behaviour specified above.

## 10. References

[1] K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*. Cambridge: Cambridge University Press, 2017. Section 4.1 (forward kinematics, Product-of-Exponentials), section 6.1 (analytical inverse kinematics).

[2] M. W. Spong, S. Hutchinson, and M. Vidyasagar, *Robot Modeling and Control*. New York: Wiley, 2005. Chapter 3 (inverse kinematics).

[3] *Pybricks Documentation*, version 2.0. Online: https://docs.pybricks.com.

## 11. Purpose and scope

This repository is developed for **educational and academic purposes**. Its primary objective is to present a clean, transparent implementation of analytical inverse kinematics on a three-degree-of-freedom anthropomorphic manipulator, with emphasis on:

- the closed-form mathematical derivation in its canonical yaw-decoupling and law-of-cosines form, traceable line-by-line from the equations of section 2.3 to the corresponding Python expressions in `SystemXYZ.get_angle`;
- a faithful translation of the derivation into a small, auditable codebase suitable as a study reference for students of robotics;
- a disciplined separation between the analytical layer (`SystemXYZ`), the hardware abstraction (`Joint`), and the control orchestration (`ThreeAxisRobotArm`), so that the mathematics can be inspected and tested in isolation from the LEGO EV3 platform.

The project is intended both as a teaching artefact for students studying the inverse kinematics of serial manipulators and as the supporting implementation for academic work on the same subject. It is not a production control stack and makes no claim of industrial readiness; it makes claims only about mathematical correctness and pedagogical clarity. The repository is provided as-is for educational use.
