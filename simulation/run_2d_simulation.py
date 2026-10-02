"""
Autonomous Navigation Simulation Runner with Live Telemetry
Runs closed-loop simulation of differential-drive robot navigated by the Neuromorphic Controller.
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure core and simulation modules are in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.snn_controller import NeuromorphicRobotController, SNNConfig
from core.lif_neuron import LIFParameters
from core.taox_synapse import TaOxParameters
from core.telemetry import TelemetryLogger
from simulation.arena import Arena
from simulation.robot_sim import SimulatedRobot


def run_navigation_simulation(
    sim_duration_sec: float = 30.0,
    control_rate_hz: float = 20.0,
    output_dir: str = "simulation_results"
) -> None:
    """
    Run closed-loop neuromorphic navigation simulation.
    
    Args:
        sim_duration_sec: Total duration to simulate (seconds)
        control_rate_hz: Decision loop frequency (Hz)
        output_dir: Folder to save trajectory and telemetry figures
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("STARTING NEUROMORPHIC ROBOT NAVIGATION SIMULATION")
    print(f"Simulation Duration: {sim_duration_sec}s | Control Loop: {control_rate_hz} Hz")
    print("Brain Architecture: LIF Neurons + TaOx Memristors with Online STDP")
    print("=" * 70)

    # Initialize environment and robot
    arena = Arena(width=10.0, height=10.0)
    robot = SimulatedRobot(x=1.8, y=1.5, theta=0.2)

    # Initialize neuromorphic controller and telemetry
    snn_config = SNNConfig()
    controller = NeuromorphicRobotController(config=snn_config)
    telemetry = TelemetryLogger(controller)

    control_period_sec = 1.0 / control_rate_hz
    control_period_ms = control_period_sec * 1000.0
    total_steps = int(sim_duration_sec * control_rate_hz)

    # Initial command
    cmd_v = 0.15
    cmd_omega = 0.0

    for step_idx in range(total_steps):
        t_sec = step_idx * control_period_sec

        # 1. Robot movement step
        robot.step(cmd_v, cmd_omega, control_period_sec)

        # 2. LiDAR scan acquisition (mimicking ROS /scan topic)
        scan = robot.scan_lidar(arena)

        # 3. Step neuromorphic brain controller over control_period_ms (e.g. 50 ms)
        # Each millisecond updates LIF equations, checks spikes, applies STDP updates
        steps_in_cycle = int(control_period_ms / snn_config.dt)
        for sub_step in range(steps_in_cycle):
            t_ms = (step_idx * control_period_ms) + (sub_step * snn_config.dt)
            pre_spikes, post_spikes = controller.step_ms(scan.d_left, scan.d_front, scan.d_right)
            
            # Subsample telemetry logging (e.g. every 5 ms)
            if sub_step % 5 == 0:
                telemetry.record_step(
                    t_ms=t_ms,
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    v_lin=cmd_v,
                    omega=cmd_omega,
                    d_left=scan.d_left,
                    d_front=scan.d_front,
                    d_right=scan.d_right
                )

        # 4. Decode accumulated motor spikes into new velocity commands
        motor_cmd = controller.decode_motor_commands()
        cmd_v = motor_cmd.linear_x
        cmd_omega = motor_cmd.angular_z

        if step_idx % int(control_rate_hz * 2) == 0:
            print(f"[{t_sec:5.1f}s] Pose=({robot.x:.2f}, {robot.y:.2f}) | "
                  f"Scan: L={scan.d_left:.2f}m F={scan.d_front:.2f}m R={scan.d_right:.2f}m | "
                  f"Cmd: v={cmd_v:.2f}m/s ω={cmd_omega:.2f}rad/s | "
                  f"Conductances: {[round(s.g, 3) for s in controller.synapses.values()]}")

    print("=" * 70)
    print("SIMULATION COMPLETED. Generating diagnostics and telemetry plots...")

    # Plot 1: Full 4-Panel Neuromorphic Telemetry Dashboard
    telemetry_path = os.path.join(output_dir, "neuromorphic_telemetry.png")
    telemetry.save_plot(telemetry_path)

    # Plot 2: 2D Arena & Robot Trajectory Map
    traj_path = os.path.join(output_dir, "robot_trajectory.png")
    plot_arena_trajectory(arena, robot, traj_path)

    print(f"Diagnostics saved successfully:\n  - Telemetry Dashboard: {telemetry_path}\n  - Trajectory Map: {traj_path}")


def plot_arena_trajectory(arena: Arena, robot: SimulatedRobot, save_path: str) -> None:
    """Generate top-down 2D map of arena with obstacles and robot trajectory."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, arena.width)
    ax.set_ylim(0, arena.height)
    ax.set_aspect('equal')
    ax.set_title("Autonomous Robot Navigation in 2D Arena\nGuided by Neuromorphic SNN Controller", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    # Draw walls and obstacles
    for wall in arena.walls:
        ax.plot([wall.x1, wall.x2], [wall.y1, wall.y2], 'k-', linewidth=2.5)

    # Draw trajectory
    traj = np.array(robot.trajectory)
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2.0, label='Robot Trajectory', alpha=0.85)

    # Draw start and finish poses
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start Position')
    ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label='End Position')

    # Draw heading arrow at final position
    arrow_len = 0.5
    ax.arrow(
        robot.x, robot.y,
        arrow_len * np.cos(robot.theta), arrow_len * np.sin(robot.theta),
        head_width=0.2, head_length=0.2, fc='red', ec='red'
    )

    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuromorphic Robot Simulation")
    parser.add_argument("--duration", type=float, default=35.0, help="Simulation duration in seconds")
    parser.add_argument("--rate", type=float, default=20.0, help="Control loop rate in Hz")
    parser.add_argument("--output_dir", type=str, default="simulation_results", help="Output directory")
    args = parser.parse_args()

    run_navigation_simulation(
        sim_duration_sec=args.duration,
        control_rate_hz=args.rate,
        output_dir=args.output_dir
    )
