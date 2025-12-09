"""
BE 124 – Lab 2.1: Simulating HILO with Gait Phase-Based Torque Control

This is a toy CMA script:
- Parameters: [peak torque (Nm), stance phase of peak (%)]
- Objective: fake stride speed (higher is better)
- CMA: minimizes cost = -stride_speed
- Output: single plot showing evolution of samples and means over generations

Last updated: December 8, 2025 by Anway Pimpalkar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# For reproducibility
np.random.seed(103)

# -------------------------
# CMA config
# -------------------------
N          = 2      # [peak torque (Nm), peak timing (% stance)]
LAMBDA     = 6      # offspring per generation
MU         = 3      # parents
GENERATIONS = 4

# Bounds: torque (Nm), timing (% stance)
PEAK_MIN, PEAK_MAX = 2.0, 20.0   # never sample below 2 Nm
OFFS_MIN, OFFS_MAX = 0.0, 100.0

# Initial mean and diagonal std devs
# Start mid-stance and moderate torque so evolution has room to move.
M_INIT   = np.array([6.0, 60.0], dtype=float)   # [peakNm, offset%]
SIGMA_INIT = np.array([6.0, 25.0], dtype=float)


def clamp_params(x):
    """Clamp [peakNm, offset%] into safe ranges."""
    x = np.array(x, dtype=float)
    x[0] = np.clip(x[0], PEAK_MIN, PEAK_MAX)
    x[1] = np.clip(x[1], OFFS_MIN, OFFS_MAX)
    return x


# -------------------------------------------------------------------
# Toy "stride speed" objective – replace this in Lab 2.2 with real data
# -------------------------------------------------------------------
def fake_stride_speed(peakNm, offsetPct):
    """
    Hidden objective: prefer late-ish peaks and high torque, with some noise.
    Returns a "stride speed" (higher is better).
    """
    # Preference for mid/late stance peak (e.g., around ~80%)
    term_offset  = -0.01 * (offsetPct - 80.0)**2

    # Preference for high torque around 18 Nm
    term_torque  = -0.02 * (peakNm - 18.0)**2

    # Small bonus for higher torque (pull upward)
    term_bonus   = +0.03 * peakNm

    # Soft safety penalty above 20 Nm
    term_safety  = -0.003 * max(0, peakNm - 20.0)**2

    base  = 2.0 + term_offset + term_torque + term_bonus + term_safety
    noise = 0.05 * np.random.randn()
    return base + noise


def evaluate_candidate(peakNm, offsetPct):
    """CMA minimizes cost, so we return negative stride speed."""
    speed = fake_stride_speed(peakNm, offsetPct)
    return -speed


# -------------------------------------------------------------------
# CMA core – this is the function you will re-use in Lab 2.2
# -------------------------------------------------------------------
def run_cma():
    """
    Runs a simple diagonal-CMA-style search on the toy objective.

    Returns:
        gen_samples: list of [lambda, 2] arrays of samples per generation
        gen_means:   list of [2] arrays (mean peakNm, mean offset%)
        gen_sigmas:  list of [2] arrays (sigma for torque, sigma for timing)
        gen_costs:   list of [lambda] arrays of costs per generation
    """
    m     = M_INIT.copy()
    sigma = SIGMA_INIT.copy()

    gen_means   = []
    gen_sigmas  = []
    gen_samples = []
    gen_costs   = []

    for gen in range(GENERATIONS):
        #################################
        ### TODO: Complete the implementation 
        #################################

    return gen_samples, gen_means, gen_sigmas, gen_costs


# -------------------------------------------------------------------
# Run CMA and plot the evolution in parameter space
# -------------------------------------------------------------------
gen_samples, gen_means, gen_sigmas, gen_costs = run_cma()

colors = plt.cm.viridis(np.linspace(0.1, 0.9, GENERATIONS))

fig, ax = plt.subplots(figsize=(7, 5))

for g in range(GENERATIONS):
    S   = gen_samples[g]   # [lambda, 2]
    m_g = gen_means[g]     # [peakNm, offset%]
    s_g = gen_sigmas[g]    # [sigma_peakNm, sigma_offset]
    c   = colors[g]

    # Scatter candidates for this generation
    ax.scatter(
        S[:, 1],     # x = stance phase (%)
        S[:, 0],     # y = peak torque (Nm)
        color=c,
        alpha=0.8,
        edgecolor="k",
        s=60,
        label=f"Gen {g+1} samples" if g == 0 else None,
    )

    # Draw 2σ ellipse around the mean (visualizing the search region)
    width  = 4 * s_g[1]    # 2σ on each side in x
    height = 4 * s_g[0]    # 2σ on each side in y
    ell = Ellipse(
        (m_g[1], m_g[0]),  # center at [offset%, peakNm]
        width=width,
        height=height,
        angle=0,
        edgecolor=c,
        facecolor="none",
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(ell)

    # Mean marker
    ax.plot(
        m_g[1], m_g[0],
        marker="x",
        markersize=10,
        mew=2.5,
        color=c
    )

# Connect mean points across generations to show trajectory
means_arr = np.stack(gen_means, axis=0)  # [gen, 2]
ax.plot(
    means_arr[:, 1],  # stance %
    means_arr[:, 0],  # peak Nm
    linestyle="--",
    color="black",
    linewidth=1.5,
    label="Mean trajectory"
)

# Formatting
ax.set_xlim(0, 100)
ax.set_ylim(0, 20)

ax.set_xlabel("Stance phase of peak torque (%)")
ax.set_ylabel("Peak torque (Nm)")
ax.tick_params(axis='both', which='major')

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("cma_evolution_param_space.svg", dpi=300, transparent=True)
plt.show()