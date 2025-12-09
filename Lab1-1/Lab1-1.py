"""
BE 124 – Lab 1.1: Proportional Torque Controller

This script:
- Sets up the Biomotum Spark in proportional torque mode
- Runs through a list of k values (scaling factors on stance torque setpoints)
- Lets the participant walk for DWELL_SECONDS at each k

Last updated: December 8, 2025 by Anway Pimpalkar
"""

from remoteSparkCommands import commands as Spark
import time

#################################
### Config: baseline setpoints ###
#################################

# Baseline torque setpoints in Nm (k = 1.0)
leftStance_base  = 1.0   # Left stance setpoint (Nm)
leftSwing_base   = 0.0   # Left swing setpoint (Nm)
rightStance_base = 1.0   # Right stance setpoint (Nm)
rightSwing_base  = 0.0   # Right swing setpoint (Nm)

#################################
### TODO: Complete the parameters
#################################

# k values to test (scale factor on stance setpoints)
K_VALUES = []

DWELL_SECONDS = 20        # time to walk for each k (seconds)
RAMP_UP_STEPS = 1         # ramp-up period in steps

#################################
### Setup & calibration #########
#################################

print("Clearing any existing commands...")
Spark.clearCommands()  # Clear any existing commands in the buffer.

# Set FSR thresholds
FSR_THRESHOLD = 0.3       # 30% threshold on both legs (0–1)
print(f"Setting FSR thresholds to {FSR_THRESHOLD*100:.0f}%")
Spark.FSRthresholds(FSR_THRESHOLD, FSR_THRESHOLD)
time.sleep(1.0)

# Set initial torque setpoints (k = 1.0 baseline)
print("Setting initial torque setpoints (baseline k = 1.0)")
# Argument order: Spark.torqueSetpoints(leftStance, leftSwing, rightStance, rightSwing)
Spark.torqueSetpoints(leftStance_base, leftSwing_base,
                      rightStance_base, rightSwing_base)
time.sleep(5.0)  # Allow a bit of time to ramp up

# Dynamic FSR calibration while user is walking
print("Recalibrating FSRs (walk during this period)...")
Spark.dynamicCalibration()
time.sleep(10.0)  # Allow time for calibration

# Set ramp-up period in steps
print(f"Setting ramp-up period to {RAMP_UP_STEPS} step(s)")
Spark.rampUpPeriod(RAMP_UP_STEPS)
time.sleep(1.0)

# Turn on motors
print("Turning motors ON")
Spark.motorsOn()
time.sleep(2.0)

#################################
### Loop over k values ##########
#################################

for i, k in enumerate(K_VALUES, start=1):
    # Scale stance torques, keep swings at baseline
    leftStance_k  = leftStance_base  * k
    rightStance_k = rightStance_base * k
    leftSwing_k   = leftSwing_base
    rightSwing_k  = rightSwing_base

    # Clip to safe range if needed (0–20 Nm)
    leftStance_k  = max(0.0, min(20.0, leftStance_k))
    rightStance_k = max(0.0, min(20.0, rightStance_k))

    print("\n" + "=" * 70)
    print(f"Condition {i}/{len(K_VALUES)}: k = {k:.2f}")
    print(f"  Left stance  = {leftStance_k:.2f} Nm")
    print(f"  Right stance = {rightStance_k:.2f} Nm")
    print("Sending scaled torque setpoints...")

    # Apply scaled torque setpoints
    Spark.torqueSetpoints(leftStance_k, leftSwing_k,
                          rightStance_k, rightSwing_k)

    # Let the participant walk for DWELL_SECONDS at this assistance level
    print(f"Hold this level and walk for ~{DWELL_SECONDS} s...")
    time.sleep(DWELL_SECONDS)

#################################
### Shutdown ####################
#################################

print("\nCompleted all k values. Zeroing setpoints and turning motors OFF...")

# Set torque setpoints to zero
Spark.torqueSetpoints(0, 0, 0, 0)
time.sleep(3.0)

# Turn off the motors
print("Turning off the motors")
Spark.motorsOff()
time.sleep(2.0)

print("Done.")