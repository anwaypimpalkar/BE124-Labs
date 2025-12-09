"""
BE 124 – Lab 2.2: Implementing HILO with gait phase-based torque control

This script:
  - Connects to the Biomotum Spark (via remoteSparkCommands).
  - Connects to an Arduino (IMU) over serial by auto-detecting manufacturer == "Arduino".
  - Uses a simple CMA loop to tune [T_peak (Nm), phi_peak (% stance)].
  - For each candidate:
      * Programs a phase-based torque profile on the Spark.
      * Lets the participant walk for TRIAL_SECONDS.
      * Reads stride-speed estimates from the Arduino over serial.
      * Uses the average stride speed as the objective (higher is better).

Last updated: December 8, 2025 by Anway Pimpalkar
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from remoteSparkCommands import commands as Spark

import serial
import serial.tools.list_ports as list_ports

# ---------------------------------------------------------
# General configuration
# ---------------------------------------------------------

# Spark config (same style as Lab 1.2)
LEFT_STANCE_NM  = 1.50
LEFT_SWING_NM   = 0.00
RIGHT_STANCE_NM = 1.50
RIGHT_SWING_NM  = 0.00

LEFT_FSR_THR  = 0.30
RIGHT_FSR_THR = 0.30
RAMP_UP_STEPS = 1

DWELL_SECONDS = 30.0    # walking time per CMA candidate
REST_BETWEEN  = 5.0     # optional rest between candidates

LEFT, RIGHT = 0, 1      # leg IDs

# CMA configuration (same as Lab 2.1)
N           = 2          # [peakNm, phi_peak%]
LAMBDA      = 6          # offspring per generation
MU          = 3          # parents
GENERATIONS = 4

# Bounds: torque (Nm), timing (% stance)
PEAK_MIN, PEAK_MAX = 0.0, 23.0
PHI_MIN,  PHI_MAX  = 0.0, 100.0

# Initial mean and step sizes
M_INIT     = np.array([6.0, 60.0], dtype=float)   # [T_peak, phi_peak]
SIGMA_INIT = np.array([6.0, 25.0], dtype=float)   # [σ_torque, σ_phase]


# ---------------------------------------------------------
# Arduino / IMU helpers
# ---------------------------------------------------------

def find_arduino_port():
    """
    Scan serial ports and return the device path whose manufacturer contains 'Arduino'.
    Raises RuntimeError if not found.
    """
    ports = list_ports.comports()
    for p in ports:
        if p.manufacturer and "Arduino" in p.manufacturer:
            return p.device
    raise RuntimeError("No Arduino device found (manufacturer='Arduino'). "
                       "Check connection and that the board is powered.")


def open_arduino_serial(baudrate=115200, timeout=1.0):
    """
    Open serial connection to the Arduino IMU.
    """
    port = find_arduino_port()
    print(f"[IMU] Connecting to Arduino on {port} …")
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    time.sleep(2.0)  # allow time for Arduino reset
    print("[IMU] Serial connection established.")
    return ser


def measure_stride_speed(ser, trial_seconds=DWELL_SECONDS):
    """
    Read stride-speed values from Arduino over serial for trial_seconds,
    return the average speed. Expects Arduino to print one float per line.
    """
    # Clear any old data
    ser.reset_input_buffer()

    speeds = []
    t_start = time.time()
    print(f"[IMU] Measuring stride speed for ~{trial_seconds:.1f} s…")

    while time.time() - t_start < trial_seconds:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                try:
                    val = float(line)
                    speeds.append(val)
                    print(f"[IMU] Received stride speed: {val:.3f}")
                except ValueError:
                    # Non-numeric line (e.g., debug text) – ignore
                    print(f"[IMU] Non-numeric line: {line}")
        except serial.SerialException as e:
            print(f"[IMU] Serial error: {e}")
            break

        time.sleep(0.05)

    if len(speeds) == 0:
        print("[IMU] No valid stride-speed values received; returning 0.0")
        return 0.0

    mean_speed = float(np.mean(speeds))
    print(f"[IMU] Mean stride speed over trial: {mean_speed:.3f}")
    return mean_speed


# ---------------------------------------------------------
# Spark / phase-based torque helpers
# ---------------------------------------------------------

def clamp_params(x):
    """Clamp [T_peak, phi_peak%] to lab-specified bounds, and round T_peak to integer Nm."""
    x = np.asarray(x, dtype=float)
    # Peak torque: integer between 0 and 23 Nm
    x[0] = float(np.clip(np.round(x[0]), PEAK_MIN, PEAK_MAX))
    # Peak timing: 0–100% stance
    x[1] = float(np.clip(x[1], PHI_MIN, PHI_MAX))
    return x


def apply_phase_profile(T_peak, phi_peak):
    """
    Program a 3-node gait-phase torque profile on both legs:

        (0%,    0 Nm)
        (phi_peak%, T_peak Nm)
        (100%,  0 Nm)

    The Spark's internal spline will interpolate between these control points.
    """
    # Ensure parameters are clamped and rounded as required
    T_peak, phi_peak = clamp_params([T_peak, phi_peak])

    x_nodes = [0.0, phi_peak, 100.0]   # stance phase (%)
    y_nodes = [0.0, T_peak,    0.0]    # torque (Nm)

    print(f"[Spark] Applying profile: T_peak={T_peak:.1f} Nm at phi_peak={phi_peak:.1f}%")
    Spark.nodeLocation(int(LEFT),  x_nodes, y_nodes)
    Spark.nodeLocation(int(RIGHT), x_nodes, y_nodes)


def spark_setup_and_calibrate():
    """
    One-time Spark setup:
      - Clear commands
      - Set FSR thresholds
      - Set baseline torque setpoints
      - Perform dynamic FSR calibration (with user walking)
      - Set ramp-up period
      - Turn motors ON
    """
    print("[Spark] Clearing command buffer…")
    Spark.clearCommands()

    print(f"[Spark] Setting FSR thresholds: L={LEFT_FSR_THR:.2f}, R={RIGHT_FSR_THR:.2f}")
    Spark.FSRthresholds(float(LEFT_FSR_THR), float(RIGHT_FSR_THR))
    time.sleep(0.5)

    print("[Spark] Setting baseline torque setpoints")
    Spark.torqueSetpoints(LEFT_STANCE_NM, LEFT_SWING_NM,
                          RIGHT_STANCE_NM, RIGHT_SWING_NM)
    time.sleep(1.5)

    print("[Spark] ** Begin walking now for dynamic FSR calibration **")
    Spark.dynamicCalibration()
    time.sleep(10.0)

    print(f"[Spark] Setting ramp-up period to {RAMP_UP_STEPS} step(s)…")
    Spark.rampUpPeriod(int(RAMP_UP_STEPS))
    time.sleep(0.5)

    print("[Spark] Turning motors ON…")
    Spark.motorsOn()
    time.sleep(1.0)


def spark_shutdown():
    """Zero torque setpoints and switch motors OFF."""
    print("\n[Spark] Zeroing setpoints and turning motors OFF for safety…")
    try:
        Spark.torqueSetpoints(0, 0, 0, 0)
    except Exception as e:
        print(f"[Spark] torqueSetpoints(0,0,0,0) failed: {e}")
    time.sleep(0.5)
    try:
        Spark.motorsOff()
    except Exception as e:
        print(f"[Spark] motorsOff() failed: {e}")
    print("[Spark] Shutdown complete.")


# ---------------------------------------------------------
# CMA core – same structure as Lab 2.1, but objective uses IMU + Spark
# ---------------------------------------------------------

def run_cma(objective_fn):
    """
    Run a simple CMA loop.

    objective_fn(T_peak, phi_peak) -> stride_speed (higher = better)

    Returns:
        gen_samples: list of [λ, 2] arrays (samples per generation)
        gen_means:   list of [2] arrays (mean per generation)
        gen_costs:   list of [λ] arrays (cost per sample)
    """
    m     = M_INIT.copy()
    sigma = SIGMA_INIT.copy()

    gen_samples = []
    gen_means   = []
    gen_costs   = []

    for gen in range(GENERATIONS):
       #################################
        ### TODO: Complete the implementation 
        #################################

    return gen_samples, gen_means, gen_costs


# ---------------------------------------------------------
# Simple evolution plot (optional but nice for the lab report)
# ---------------------------------------------------------

def plot_cma_evolution(gen_samples, gen_means, filename="cma_evolution_param_space.svg"):
    FONTSIZE = 16
    generations = len(gen_samples)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, generations))

    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter samples by generation
    for g in range(generations):
        S = gen_samples[g]
        c = colors[g]
        ax.scatter(
            S[:, 1],      # x = stance phase %
            S[:, 0],      # y = peak torque Nm
            color=c,
            alpha=0.8,
            edgecolor="k",
            s=60,
            label=f"Gen {g+1}" if g == 0 else None,
        )

    # Mean trajectory
    means_arr = np.stack(gen_means, axis=0)
    ax.plot(
        means_arr[:, 1],   # stance phase
        means_arr[:, 0],   # peak torque
        linestyle="--",
        color="black",
        linewidth=1.5,
        marker="x",
        markersize=8,
        label="Mean (per generation)",
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 23)
    ax.set_xlabel("Stance phase of peak torque (%)", fontsize=FONTSIZE)
    ax.set_ylabel("Peak torque (Nm)", fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True)
    plt.show()
    print(f"[CMA] Saved evolution plot to {filename}")


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

if __name__ == "__main__":
    imu_serial = None
    try:
        # 1) Connect to Arduino IMU
        imu_serial = open_arduino_serial(baudrate=115200, timeout=1.0)

        # 2) Setup and calibrate the Spark
        spark_setup_and_calibrate()

        # 3) Define objective that uses Spark + IMU
        def objective_with_spark_and_imu(T_peak, phi_peak):
            # Apply phase-based profile on Spark
            apply_phase_profile(T_peak, phi_peak)
            time.sleep(1.0)  # small settling time (ramp + user adjustment)
            # Measure stride speed from IMU
            return measure_stride_speed(imu_serial, trial_seconds=DWELL_SECONDS)

        # 4) Run CMA
        gen_samples, gen_means, gen_costs = run_cma(objective_with_spark_and_imu)

        # 5) Optional: plot evolution in parameter space
        plot_cma_evolution(gen_samples, gen_means)

        # Print final mean for log / report
        final_mean = gen_means[-1]
        print("\n[CMA] Final mean parameters:")
        print(f"     T_peak = {final_mean[0]:.2f} Nm")
        print(f"     phi_peak = {final_mean[1]:.2f} % stance")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user (Ctrl+C).")

    except Exception as e:
        print(f"\n[Main] ERROR: {e}")

    finally:
        # Ensure Spark is made safe
        try:
            spark_shutdown()
        except Exception as e:
            print(f"[Main] spark_shutdown() failed: {e}")

        # Close IMU serial
        if imu_serial is not None and imu_serial.is_open:
            imu_serial.close()
            print("[IMU] Serial connection closed.")