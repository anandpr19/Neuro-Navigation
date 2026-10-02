# Neuro-Navigation: Brain-Inspired Computing System for Autonomous Robot Control

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-orange.svg)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-purple.svg)](https://gazebosim.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous robotic navigation system that replaces conventional deterministic control algorithms with a biological paradigm: a **Spiking Neural Network (SNN)** featuring **Leaky Integrate-and-Fire (LIF) neurons** and nanoscale **Tantalum Oxide ($\text{TaO}_x$) memristive synapses** governed by online **Spike-Timing-Dependent Plasticity (STDP)**.

The system runs in two flexible execution environments:
1. **Standalone 2D Kinematic Simulator** (Pure Python, dependency-light, live telemetry diagnostics).
2. **ROS 2 & Gazebo 3D Environment** (Native ROS 2 Jazzy control node subscribing to `/scan` and publishing `/cmd_vel`).

---

## 📑 Table of Contents

- [1. Key Features](#1-key-features)
- [2. System Architecture & Mathematical Foundations](#2-system-architecture--mathematical-foundations)
  - [2.1 Signal Flow Diagram](#21-signal-flow-diagram)
  - [2.2 Leaky Integrate-and-Fire (LIF) Neuron Model](#22-leaky-integrate-and-fire-lif-neuron-model)
  - [2.3 $\text{TaO}_x$ Memristor Dynamics & STDP Plasticity](#23-taox-memristor-dynamics--stdp-plasticity)
  - [2.4 Sensory Encoding & Motor Decoding Codecs](#24-sensory-encoding--motor-decoding-codecs)
- [3. Repository Structure](#3-repository-structure)
- [4. Getting Started & Installation](#4-getting-started--installation)
  - [4.1 Prerequisites](#41-prerequisites)
  - [4.2 Python Environment Setup](#42-python-environment-setup)
- [5. Execution & Usage](#5-execution--usage)
  - [5.1 Standalone 2D Simulation](#51-standalone-2d-simulation)
  - [5.2 Running Pytest Unit Tests](#52-running-pytest-unit-tests)
  - [5.3 ROS 2 & Gazebo 3D Simulation](#53-ros-2--gazebo-3d-simulation)
- [6. Telemetry & Diagnostics Output](#6-telemetry--diagnostics-output)
- [7. Hyperparameter Tuning Reference](#7-hyperparameter-tuning-reference)
- [8. License](#8-license)

---

## 1. Key Features

- **Biophysical SNN Controller**: Closed-loop navigation driven by LIF spike integration and Poisson rate encoding.
- **Nanoscale Memristive Synapses**: Synthetic $\text{TaO}_x$ memristors with bounded conductances $G \in [G_{min}, G_{max}] = [0.001\text{ S}, 1.0\text{ S}]$.
- **Online STDP Learning**: Asymmetric Spike-Timing-Dependent Plasticity adaptively adjusts synaptic weights based on pre/post spike deltas ($\Delta t = t_{post} - t_{pre}$).
- **Dual Execution Engines**:
  - **Fast 2D Standalone Simulator**: Pure Python raycasting simulator with obstacle arenas and instant visual plot generation.
  - **Native ROS 2 Node**: Fully compatible with ROS 2 Jazzy Jalisco, Gazebo Harmonic, and TurtleBot3 Burger hardware/simulators (`Twist` & `TwistStamped`).
- **Comprehensive Diagnostic Suite**: Generates 4-panel time-series dashboards capturing sensory distances, membrane potential traces $V(t)$, neural spike rasters, and memristor conductance evolution $G(t)$.

---

## 2. System Architecture & Mathematical Foundations

### 2.1 Signal Flow Diagram

```
       ┌────────────────────────┐
       │   LiDAR Sensor Scan    │
       │ (/scan or 2D Raycast)  │
       └───────────┬────────────┘
                   │  Distance (d_left, d_front, d_right)
                   ▼
       ┌────────────────────────┐
       │  Sensory Rate Encoder  │  f(d) = f_max * (1 - min(d, d_max)/d_max)
       └───────────┬────────────┘
                   │  Poisson Pre-Synaptic Spikes
                   ▼
       ┌────────────────────────┐
       │ Pre-Synaptic LIF Array │  (S_LEFT, S_FRONT, S_RIGHT)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐    STDP Feedback
       │ TaOx Memristor Synapses│ ◄──────────────────┐
       │   Weight Conductance G │ ─────────────────┐ │
       └───────────┬────────────┘                  │ │
                   │ Synaptic Current I(t)         │ │
                   ▼                               │ │
       ┌────────────────────────┐                  │ │
       │ Post-Synaptic LIF Array│                  │ │
       │   (M_FWD, M_LEFT, M_R) │ ─────────────────┼─┘
       └───────────┬────────────┘  Post Spikes t_post
                   │  Action Potentials
                   ▼
       ┌────────────────────────┐
       │  Motor Spike Decoder   │  Sliding Window ΔT = 100ms
       └───────────┬────────────┘
                   │  Twist (linear v, angular ω)
                   ▼
       ┌────────────────────────┐
       │ Robot Base Kinematics  │
       │ (/cmd_vel & Gazebo Gz) │
       └────────────────────────┘
```

### 2.2 Leaky Integrate-and-Fire (LIF) Neuron Model

Membrane potentials $V(t)$ integrate incoming synaptic currents according to the differential equation:

$$\tau_m \frac{dV(t)}{dt} = -(V(t) - V_{rest}) + R_m \cdot I(t)$$

| Parameter | Symbol | Nominal Value | Description |
| :--- | :--- | :--- | :--- |
| **Resting Potential** | $V_{rest}$ | $-70.0\text{ mV}$ | Baseline membrane voltage |
| **Threshold Potential** | $V_{th}$ | $-55.0\text{ mV}$ | Action potential trigger threshold |
| **Reset Potential** | $V_{reset}$ | $-70.0\text{ mV}$ | Voltage post-spike emission |
| **Membrane Time Constant** | $\tau_m$ | $20.0\text{ ms}$ | Passive voltage decay constant |
| **Refractory Period** | $\tau_{ref}$ | $2.0\text{ ms}$ | Clamping period post-spike |

### 2.3 $\text{TaO}_x$ Memristor Dynamics & STDP Plasticity

Synaptic connections between pre- and post-synaptic neurons are physical conductances $G$. When spikes occur at $t_{pre}$ and $t_{post}$ ($\Delta t = t_{post} - t_{pre}$), weight adaptation follows:

$$\Delta G = \begin{cases} 
A_+ \cdot \exp\left(-\frac{\Delta t}{\tau_+}\right) & \text{for } \Delta t > 0 \quad (\text{Long-Term Potentiation / LTP}) \\
-A_- \cdot \exp\left(\frac{\Delta t}{\tau_-}\right) & \text{for } \Delta t < 0 \quad (\text{Long-Term Depression / LTD})
\end{cases}$$

$$G(t + \Delta t) = \text{clip}\left(G(t) + \Delta G,\ G_{min},\ G_{max}\right)$$

* **$A_+ = A_- = 0.01\text{ S}$**: STDP learning rate amplitudes
* **$\tau_+ = \tau_- = 15.0\text{ ms}$**: Temporal STDP pairing window constants
* **$[G_{min}, G_{max}] = [0.001\text{ S}, 1.0\text{ S}]$**: Hard physical conductance boundary limits

### 2.4 Sensory Encoding & Motor Decoding Codecs

* **Sensor Rate Encoder**: LiDAR sector obstacle distance $d$ converts into spike firing rate $f(d)$:
  $$f(d) = f_{max} \cdot \left(1 - \frac{\min(d, d_{max})}{d_{max}}\right)$$
* **Motor Velocity Decoder**: Accumulated spikes in motor windows over evaluation window $\Delta T$:
  $$v = \min\left(v_{max},\ k_v \cdot \frac{\text{Count}_{fwd}}{\Delta T}\right)$$
  $$\omega = \text{clip}\left(k_\omega \cdot \frac{\text{Count}_{left} - \text{Count}_{right}}{\Delta T},\ -\omega_{max},\ \omega_{max}\right)$$

---

## 3. Repository Structure

```
Neuro-Navigation/
├── core/                           # Standalone Neuromorphic Computing Engine
│   ├── lif_neuron.py               # LIF neuron differential equation solver & state manager
│   ├── taox_synapse.py             # TaOx memristor model with STDP plasticity
│   ├── snn_controller.py           # Rate codecs, topology mapping & velocity decoder
│   └── telemetry.py                # Diagnostic telemetry logger & visualizer
├── simulation/                     # 2D Interactive Kinematic Simulator
│   ├── arena.py                    # 2D environment, obstacle walls & raycasting
│   ├── robot_sim.py                # Kinematic robot + 3-sector LiDAR scanner
│   └── run_2d_simulation.py        # Real-time simulation runner & diagnostic plotter
├── ros2_ws/                        # ROS 2 Colcon Workspace
│   └── src/
│       └── brain_robot_control/
│           ├── brain_robot_control/
│           │   └── neuromorphic_node.py  # ROS 2 Node (/scan -> SNN -> /cmd_vel)
│           ├── config/
│           │   └── controller_params.yaml # ROS 2 parameter YAML file
│           ├── launch/
│           │   ├── controller.launch.py   # Standalone ROS 2 node launch file
│           │   ├── gazebo_simulation.launch.py # Full 3D Gazebo simulation launcher
│           │   └── gazebo_headless.launch.py   # Headless VM launch script
│           └── worlds/
│               └── neuromorphic_arena.world # Gazebo SDF world description
├── scripts/
│   ├── setup_ros2_gazebo.sh        # One-click installation script for ROS 2 Jazzy & Gazebo
│   ├── build_and_run_ros.sh        # Colcon workspace builder & launch script
│   └── run_gazebo_vm.sh            # VM / VMware GPU software rendering wrapper
├── tests/                          # Pytest Automated Test Suite
│   ├── test_lif_neuron.py          # Unit tests for membrane leak & threshold reset
│   ├── test_taox_synapse.py        # Unit tests for LTP, LTD, and conductance bounds
│   └── test_snn_controller.py      # Unit tests for rate encoding & motor decoding
├── simulation_results/             # Generated telemetry graphs & trajectory plots
└── README.md                       # Project documentation
```

---

## 4. Getting Started & Installation

### 4.1 Prerequisites

- **Python 3.10+** (Required for 2D Standalone Simulator)
- **Linux Ubuntu 24.04 LTS (Noble Numbat)** or **WSL2** (Required for ROS 2 Jazzy & Gazebo Harmonic)

### 4.2 Python Environment Setup

Clone the repository and install required dependencies:

```bash
git clone https://github.com/your-username/Neuro-Navigation.git
cd Neuro-Navigation

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install numpy matplotlib pytest
```

---

## 5. Execution & Usage

### 5.1 Standalone 2D Simulation

To run the closed-loop navigation simulation in the 2D arena and generate live diagnostic plots:

```bash
python simulation/run_2d_simulation.py --duration 35.0 --rate 20.0
```

**Output Files Created**:
- `simulation_results/neuromorphic_telemetry.png`: 4-panel live telemetry dashboard.
- `simulation_results/robot_trajectory.png`: 2D top-down map of arena obstacles and robot path.

### 5.2 Running Pytest Unit Tests

To verify the mathematical accuracy of LIF neurons, $\text{TaO}_x$ STDP equations, and rate codecs:

```bash
pytest tests/ -v
```

### 5.3 ROS 2 & Gazebo 3D Simulation

#### Step 1: Install ROS 2 Jazzy & Gazebo Harmonic
Run the automated installation script:
```bash
./scripts/setup_ros2_gazebo.sh
```

#### Step 2: Build the Colcon Workspace
```bash
./scripts/build_and_run_ros.sh
```

#### Step 3: Launch 3D Simulation in Gazebo
```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

# Launch Gazebo + Neuromorphic Controller Node
ros2 launch brain_robot_control gazebo_simulation.launch.py
```

*For Virtual Machine (VMware/VirtualBox) users facing GPU graphics glitches*:
```bash
./scripts/run_gazebo_vm.sh software_gui
# Or for headless execution:
./scripts/run_gazebo_vm.sh headless
```

---

## 6. Telemetry & Diagnostics Output

When running simulations, the `TelemetryLogger` automatically samples neural parameters and outputs publication-quality visualizations:

1. **Panel 1: Sensory Input & Motor Commands**: LiDAR sector distances vs linear ($v$) and angular ($\omega$) velocity output over time.
2. **Panel 2: Motor LIF Voltage Traces**: Membrane potential trajectories $V(t)$ of post-synaptic motor neurons (`M_FWD`, `M_LEFT`, `M_RIGHT`) against $V_{th}$ and $V_{rest}$.
3. **Panel 3: Neural Spike Raster**: Firing timestamps across all pre-synaptic sensory neurons and post-synaptic motor neurons.
4. **Panel 4: $\text{TaO}_x$ Conductance Adaptation**: Real-time conductance trajectories $G(t)$ illustrating STDP plasticity weight updates.

---

## 7. Hyperparameter Tuning Reference

All core controller parameters can be customized via `SNNConfig` in [snn_controller.py](file:///D:/STUDY%20MATERIAL/DATA%20FILES/PROJECTS/Neuro-Navigation/core/snn_controller.py) or `controller_params.yaml`:

```python
@dataclass
class SNNConfig:
    f_max: float = 90.0           # Max firing frequency at close proximity (Hz)
    d_max: float = 0.70           # Sensor range cutoff distance (m)
    dt: float = 1.0               # Neural integration time step (ms)
    tau_syn: float = 6.0          # Synaptic current exponential decay time (ms)
    window_dt: float = 100.0      # Motor spike evaluation sliding window ΔT (ms)
    max_linear_vel: float = 0.22  # Maximum linear velocity (m/s)
    max_angular_vel: float = 1.50 # Maximum angular velocity (rad/s)
    k_v: float = 0.015            # Linear velocity decoding multiplier
    k_omega: float = 0.12         # Angular velocity decoding multiplier
```

---

## 8. License

This project is open-source under the [MIT License](LICENSE).
