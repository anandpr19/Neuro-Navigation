"""
Live Telemetry Logger & Visualization for Neuromorphic Brain Controller
Records membrane potentials, spike rasters, synaptic conductances, and robot velocities.
"""

from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless & server rendering
import matplotlib.pyplot as plt
import numpy as np

from .snn_controller import NeuromorphicRobotController


class TelemetryLogger:
    """
    Collects time-series diagnostics from NeuromorphicRobotController
    and generates visual telemetry plots.
    """

    def __init__(self, controller: NeuromorphicRobotController):
        self.controller = controller
        self.timestamps: List[float] = []
        
        # Membrane potentials V(t) in mV
        self.voltages: Dict[str, List[float]] = {
            n_id: [] for n_id in list(controller.sensor_ids) + list(controller.motor_ids)
        }
        
        # Synaptic conductances G(t) in Siemens
        self.conductances: Dict[str, List[float]] = {
            s_name: [] for s_name in controller.synapses.keys()
        }
        
        # Spike events: list of (time_ms, neuron_id)
        self.spike_events: List[tuple] = []
        
        # Robot velocities & sensor distances
        self.linear_velocities: List[float] = []
        self.angular_velocities: List[float] = []
        self.sensor_left: List[float] = []
        self.sensor_front: List[float] = []
        self.sensor_right: List[float] = []

    def record_step(
        self,
        t_ms: float,
        pre_spikes: Dict[str, bool],
        post_spikes: Dict[str, bool],
        v_lin: float,
        omega: float,
        d_left: float,
        d_front: float,
        d_right: float
    ) -> None:
        """Sample current state and record into telemetry logs."""
        self.timestamps.append(t_ms)

        # Record sensory neuron voltages
        for s_id, neuron in self.controller.sensor_neurons.items():
            self.voltages[s_id].append(neuron.v)
            if pre_spikes.get(s_id, False):
                self.spike_events.append((t_ms, s_id))

        # Record motor neuron voltages
        for m_id, neuron in self.controller.motor_neurons.items():
            self.voltages[m_id].append(neuron.v)
            if post_spikes.get(m_id, False):
                self.spike_events.append((t_ms, m_id))

        # Record synaptic conductances
        for s_name, syn in self.controller.synapses.items():
            self.conductances[s_name].append(syn.g)

        # Record velocities and distances
        self.linear_velocities.append(v_lin)
        self.angular_velocities.append(omega)
        self.sensor_left.append(d_left)
        self.sensor_front.append(d_front)
        self.sensor_right.append(d_right)

    def save_plot(self, output_path: str = "neuromorphic_telemetry.png") -> None:
        """
        Generate and save a 4-panel diagnostic plot:
        1. Sensory Distances & Commanded Velocities
        2. Post-Synaptic Motor Neuron Membrane Potentials V(t)
        3. Neural Spike Raster Plot
        4. TaOx Memristor Conductance Evolution G(t)
        """
        if not self.timestamps:
            return

        times = np.array(self.timestamps) / 1000.0  # Convert to seconds
        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(
            "Brain-Inspired Neuromorphic Robot Control - Live Telemetry\n"
            "TaOx Memristor STDP Plasticity & Leaky Integrate-and-Fire Dynamics",
            fontsize=14,
            fontweight="bold"
        )

        # Panel 1: Sensor Distances & Commanded Velocities
        ax0 = axes[0]
        ax0_twin = ax0.twinx()
        l1, = ax0.plot(times, self.sensor_front, 'k--', label='Front Distance (m)', alpha=0.7)
        l2, = ax0.plot(times, self.sensor_left, 'b:', label='Left Distance (m)', alpha=0.7)
        l3, = ax0.plot(times, self.sensor_right, 'r:', label='Right Distance (m)', alpha=0.7)
        ax0.set_ylabel("LiDAR Distance (m)", color="k", fontweight="semibold")
        ax0.set_ylim(0, 3.0)

        l4, = ax0_twin.plot(times, self.linear_velocities, 'g-', label='Linear Vel v (m/s)', linewidth=1.5)
        l5, = ax0_twin.plot(times, self.angular_velocities, 'm-', label='Angular Vel ω (rad/s)', linewidth=1.5)
        ax0_twin.set_ylabel("Motor Velocity (m/s, rad/s)", color="g", fontweight="semibold")
        ax0_twin.set_ylim(-2.0, 2.0)
        
        lines = [l1, l2, l3, l4, l5]
        ax0.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8, ncol=3)
        ax0.grid(True, linestyle=":", alpha=0.6)
        ax0.set_title("Sensory Input & Decoded Motor Velocity Commands", fontsize=11, fontweight="bold")

        # Panel 2: Motor Neuron Membrane Potentials V(t)
        ax1 = axes[1]
        ax1.plot(times, self.voltages["M_FWD"], 'g-', label="M_FWD V(t)", alpha=0.8)
        ax1.plot(times, self.voltages["M_LEFT"], 'b-', label="M_LEFT V(t)", alpha=0.8)
        ax1.plot(times, self.voltages["M_RIGHT"], 'r-', label="M_RIGHT V(t)", alpha=0.8)
        ax1.axhline(self.controller.lif_params.v_th, color='r', linestyle='--', label=f"V_th ({self.controller.lif_params.v_th} mV)")
        ax1.axhline(self.controller.lif_params.v_rest, color='gray', linestyle=':', label=f"V_rest ({self.controller.lif_params.v_rest} mV)")
        ax1.set_ylabel("Membrane V(t) (mV)", fontweight="semibold")
        ax1.set_ylim(-75, -50)
        ax1.legend(loc="upper right", fontsize=8, ncol=5)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.set_title("Post-Synaptic Motor Neuron LIF Dynamics", fontsize=11, fontweight="bold")

        # Panel 3: Spike Raster Plot
        ax2 = axes[2]
        all_neurons = list(self.controller.sensor_ids) + list(self.controller.motor_ids)
        neuron_y_map = {nid: idx for idx, nid in enumerate(all_neurons)}
        
        spike_times_sec = [ev[0] / 1000.0 for ev in self.spike_events]
        spike_neuron_indices = [neuron_y_map[ev[1]] for ev in self.spike_events]
        
        ax2.scatter(spike_times_sec, spike_neuron_indices, marker='|', color='navy', s=80, linewidth=1.5)
        ax2.set_yticks(range(len(all_neurons)))
        ax2.set_yticklabels(all_neurons, fontsize=9, fontweight="semibold")
        ax2.set_ylabel("Neuron ID", fontweight="semibold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.set_title("Neural Spike Raster (Sensory Pre-Synaptic & Motor Post-Synaptic)", fontsize=11, fontweight="bold")

        # Panel 4: TaOx Memristor Conductance Evolution G(t)
        ax3 = axes[3]
        for s_name, g_vals in self.conductances.items():
            ax3.plot(times, g_vals, label=f"{s_name} G(t)", linewidth=1.8)
        ax3.axhline(self.controller.taox_params.g_min, color='gray', linestyle=':', label='G_min')
        ax3.axhline(self.controller.taox_params.g_max, color='gray', linestyle='--', label='G_max')
        ax3.set_ylabel("Conductance G (S)", fontweight="semibold")
        ax3.set_xlabel("Time (seconds)", fontweight="semibold")
        ax3.legend(loc="upper right", fontsize=8, ncol=4)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.set_title("TaOx Memristor Synaptic Conductance (STDP Plasticity Adaptation)", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"[Telemetry] Saved live telemetry dashboard to: {output_path}")
