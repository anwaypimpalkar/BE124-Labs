"""
BE 124 – Lab 1.2: Gait Phase-Based Torque Controller

This script:
- Sets up the Biomotum Spark for phase-based torque control
- Applies a series of 3-node torque profiles over stance phase
- Lets the participant walk for DWELL_SECONDS with each profile

Each torque profile is defined by three control points:
    [(0, 0), (phi_peak, T_peak), (100, 0)]
where:
    phi_peak is the stance phase (%) of peak torque
    T_peak is the peak torque (Nm)

Last updated: December 8, 2025 by Anway Pimpalkar
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from remoteSparkCommands import commands as Spark  # keep alias as-is

#################################
### Config: baseline setpoints ###
#################################

# Baseline stance/swing torque setpoints in Nm
LEFT_STANCE_NM  = 1.50
LEFT_SWING_NM   = 0.00
RIGHT_STANCE_NM = 1.50
RIGHT_SWING_NM  = 0.00

# Leg IDs (adjust only if your firmware differs)
LEFT, RIGHT = 0, 1

#################################
### TODO: Complete the parameters
#################################

# Torque profiles to iterate through.
# Each entry is a list of three nodes:
#   [(0, 0), (phi_peak, T_peak), (100, 0)]
# where:
#   phi_peak is in [0, 100] (% stance)
#   T_peak  is in [0, 23]   (Nm)
TORQUE_PROFILES = [

]

DWELL_SECONDS  = 20    # time to walk for each profile (seconds)
RAMP_UP_STEPS  = 1     # ramp-up period in steps

#################################
### Helper: spline evaluation ###
#################################

def eval_peak_preserving_cubic(x_nodes, y_nodes, x_eval):
    """
    Peak-preserving two-segment cubic with zero slopes at all three nodes.
    x_nodes: [x0, x1, x2]   (stance %)
    y_nodes: [y0, y1, y2]   (torque in Nm)
    x_eval:  array of stance % values to evaluate
    """
    x0, x1, x2 = map(float, x_nodes)
    y0, y1, y2 = map(float, y_nodes)
    xe = np.asarray(x_eval, dtype=float)

    def hermite_00_01(t, ya, yb):
        # Zero-slope Hermite from (ya,0) -> (yb,0), t in [0, 1]
        # h00 = 2t^3 - 3t^2 + 1, h01 = -2t^3 + 3t^2
        return ya * (2*t**3 - 3*t**2 + 1) + yb * (-2*t**3 + 3*t**2)

    y = np.empty_like(xe)

    # Left segment [x0, x1]
    left_mask = xe <= x1
    if np.any(left_mask):
        tL = np.clip((xe[left_mask] - x0) / (x1 - x0 + 1e-12), 0, 1)
        y[left_mask] = hermite_00_01(tL, y0, y1)

    # Right segment [x1, x2]
    right_mask = xe > x1
    if np.any(right_mask):
        tR = np.clip((xe[right_mask] - x1) / (x2 - x1 + 1e-12), 0, 1)
        y[right_mask] = hermite_00_01(tR, y1, y2)

    return y

#################################
### PRE-RUN: Plot torque curves #
#################################

def preview_profiles():
    if not TORQUE_PROFILES:
        print("No torque profiles defined; skipping preview plot.")
        return

    plt.figure(figsize=(7.5, 4.8))
    x_dense = np.linspace(0, 100, 600)

    for i, nodes in enumerate(TORQUE_PROFILES, start=1):
        if len(nodes) != 3:
            print(f"Profile {i} does not have 3 control points; skipping.")
            continue

        xs = [float(nodes[j][0]) for j in range(3)]
        ys = [float(nodes[j][1]) for j in range(3)]

        # Clamp T_peak into [0, 23] Nm for safety (middle node)
        ys[1] = max(0.0, min(23.0, ys[1]))

        y_curve = eval_peak_preserving_cubic(xs, ys, x_dense)
        plt.plot(x_dense, y_curve, label=f'Profile {i}')
        plt.plot(xs, ys, 'o', ms=5)  # markers at control points

    plt.title('Phase-Based Torque Profiles\nY: torque (Nm), X: stance phase (%)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Torque (Nm)')
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.show()

#################################
### Setup & calibration #########
#################################

def setup_spark():
    print("Clearing any existing commands...")
    Spark.clearCommands()
    
    FSR_THRESHOLD  = 0.30  # 30% threshold on both legs (0–1)
    print(f"Setting FSR thresholds to {FSR_THRESHOLD*100:.0f}% on both legs")
    Spark.FSRthresholds(FSR_THRESHOLD, FSR_THRESHOLD)
    time.sleep(1.0)

    print("Setting baseline torque setpoints "
          f"L(stance={LEFT_STANCE_NM}, swing={LEFT_SWING_NM}), "
          f"R(stance={RIGHT_STANCE_NM}, swing={RIGHT_SWING_NM})")
    Spark.torqueSetpoints(LEFT_STANCE_NM, LEFT_SWING_NM,
                          RIGHT_STANCE_NM, RIGHT_SWING_NM)
    time.sleep(5.0)

    print("Recalibrating FSRs (walk during this period)...")
    Spark.dynamicCalibration()
    time.sleep(10.0)

    print(f"Setting ramp-up period to {RAMP_UP_STEPS} step(s)")
    Spark.rampUpPeriod(RAMP_UP_STEPS)
    time.sleep(1.0)

    print("Turning motors ON")
    Spark.motorsOn()
    time.sleep(2.0)

#################################
### Apply profiles ##############
#################################

def apply_profile_to_leg(leg_id, profile_nodes):
    """
    profile_nodes: [(0, 0), (phi_peak, T_peak), (100, 0)]
    """
    if len(profile_nodes) != 3:
        raise ValueError("Each profile must have exactly 3 nodes.")

    x_nodes = [float(p[0]) for p in profile_nodes]
    y_nodes = [float(p[1]) for p in profile_nodes]

    # Clamp phi into [0, 100], T_peak into [0, 23]
    x_nodes = [max(0.0, min(100.0, x)) for x in x_nodes]
    y_nodes[1] = max(0.0, min(23.0, y_nodes[1]))  # peak only

    Spark.nodeLocation(int(leg_id), x_nodes, y_nodes)

#################################
### Main execution ##############
#################################

try:
    # Optional: visualize the profiles before running on hardware
    preview_profiles()

    # Setup Spark (FSRs, calibration, ramp-up, motors on)
    setup_spark()

    # Cycle through all torque profiles
    for i, profile in enumerate(TORQUE_PROFILES, start=1):
        print("\n" + "=" * 70)
        print(f"Applying torque profile {i}/{len(TORQUE_PROFILES)}: {profile}")
        apply_profile_to_leg(LEFT,  profile)
        apply_profile_to_leg(RIGHT, profile)

        print(f"Hold this profile and walk for ~{DWELL_SECONDS} s...")
        time.sleep(DWELL_SECONDS)

    print("\nCompleted all torque profiles.")

finally:
    print("\nZeroing setpoints and turning motors OFF for safety...")
    try:
        Spark.torqueSetpoints(0, 0, 0, 0)
    except Exception as e:
        print(f"Setpoint zero failed: {e}")
    time.sleep(3.0)

    try:
        Spark.motorsOff()
    except Exception as e:
        print(f"motorsOff failed: {e}")

    print("Done.")