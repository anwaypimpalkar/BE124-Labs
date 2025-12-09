# BE 124 Exoskeleton Control Labs

Last updated: December 8, 2025 by Anway Pimpalkar

This repository contains starter code for the four modules in the Harvard University BE 124 exoskeleton control labs. The labs use the Biomotum Spark ankle exoskeleton and an external IMU (Arduino Nano 33 BLE Sense) to explore proportional control, gait phase–based control, and human-in-the-loop optimization.

## Labs Structure

The labs are completed in the following order:

1. `Lab 1.1 – Proportional Torque Control:` You will scale the biological-weight torque estimate by a gain `k` and observe how assistance magnitude changes during stance. You will complete the template script and run a set of prescribed gain values.
2. `Lab 1.2 – Gait Phase-Based Torque Control:` You will implement a spline-based stance-phase torque controller. Nine different torque profiles are provided. You will complete the script, apply each profile, collect telemetry, and evaluate how timing and magnitude affect assistance.
3. `Lab 2.1 – Simulating HILO with CMA:` This lab introduces a simplified version of Covariance Matrix Adaptation (CMA). You will implement the algorithm on a toy objective function (simulated stride speed) and generate a plot showing how the sampled candidates and CMA mean evolve over generations.
4. `Lab 2.2 – Implementing HILO on the Spark:` You will combine the CMA algorithm from Lab 2.1 with the gait phase–based control from Lab 1.2. Instead of a synthetic objective, you will measure stride speed from the Arduino IMU. CMA will propose torque parameters, the Spark will apply them, and the IMU will return stride-speed values for each trial.

## Running the Labs

1. Start a Spark session in the Biomotum GUI and enable Scripted Control.
2. Ensure the Arduino is connected and streaming stride-speed values.
3. Run the corresponding lab script from the command line or your IDE.
4. Save all Spark telemetry and IMU logs after completing each lab.
5. Include observations and required plots in your lab report.
