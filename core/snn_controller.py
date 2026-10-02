"""
Neuromorphic Spiking Neural Network (SNN) Robot Controller
Orchestrates sensor rate encoding, LIF neural integration, TaOx memristive STDP plasticity,
and motor velocity decoding for closed-loop navigation.
"""

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Tuple

from .lif_neuron import LIFNeuron, LIFParameters
from .taox_synapse import TaOxSynapse, TaOxParameters


@dataclass
class SNNConfig:
    """Configuration for neuromorphic controller and signal conversion pipelines."""
    # Sensor Rate Encoding
    f_max: float = 90.0              # Maximum firing frequency at contact (Hz)
    d_max: float = 0.70              # Maximum sensor range cutoff (meters, tuned for pillar gaps)
    dt: float = 1.0                  # Simulation integration time step (ms)
    
    # Synaptic Dynamics
    tau_syn: float = 6.0             # Synaptic current exponential decay time constant (ms)
    
    # Motor Velocity Decoding
    window_dt: float = 100.0         # Evaluation sliding window ΔT (ms)
    max_linear_vel: float = 0.22     # Maximum forward linear velocity (m/s, TurtleBot3 limit)
    max_angular_vel: float = 1.50    # Maximum angular velocity (rad/s)
    k_v: float = 0.015               # Linear velocity scale constant per spike count
    k_omega: float = 0.12            # Angular velocity scale constant per spike difference
    
    # Baseline forward excitation current (nA) when path is clear (rheobase is 15 nA)
    baseline_forward_current: float = 26.0


@dataclass
class MotorCommand:
    """Decoded velocity commands for the robot base."""
    linear_x: float     # m/s
    angular_z: float    # rad/s


class NeuromorphicRobotController:
    """
    Closed-loop neuromorphic robot controller combining:
    1. Sensor Rate Encoder: LaserScan sectors -> Pre-synaptic spike trains
    2. TaOx Memristor Synaptic Array: Adaptive weights with STDP
    3. Post-synaptic LIF Neurons: Membrane potential integration & action potentials
    4. Motor Velocity Decoder: Spike rate accumulation -> Twist (v, omega)
    """

    def __init__(
        self,
        config: Optional[SNNConfig] = None,
        lif_params: Optional[LIFParameters] = None,
        taox_params: Optional[TaOxParameters] = None
    ):
        self.config = config or SNNConfig()
        self.lif_params = lif_params or LIFParameters()
        self.taox_params = taox_params or TaOxParameters()
        self.sim_time_ms: float = 0.0

        # Define Sensory Pre-synaptic Neurons (Front, Left, Right)
        self.sensor_ids = ["S_LEFT", "S_FRONT", "S_RIGHT"]
        self.sensor_neurons: Dict[str, LIFNeuron] = {
            s_id: LIFNeuron(s_id, self.lif_params) for s_id in self.sensor_ids
        }

        # Define Motor Post-synaptic Neurons (Forward, TurnLeft, TurnRight)
        self.motor_ids = ["M_FWD", "M_LEFT", "M_RIGHT"]
        self.motor_neurons: Dict[str, LIFNeuron] = {
            m_id: LIFNeuron(m_id, self.lif_params) for m_id in self.motor_ids
        }

        # Persistent synaptic currents with exponential decay
        self.synaptic_currents: Dict[str, float] = {m_id: 0.0 for m_id in self.motor_ids}

        # Define Synaptic Cross-Connections (Pre -> Post)
        # Avoidance topology with balanced initial conductances:
        # S_LEFT -> M_RIGHT (obstacle on left evokes right turn)
        # S_RIGHT -> M_LEFT (obstacle on right evokes left turn)
        # S_FRONT -> M_LEFT & M_RIGHT (obstacle in front evokes balanced avoidance)
        self.synapses: Dict[str, TaOxSynapse] = {
            "syn_L_to_R": TaOxSynapse("syn_L_to_R", "S_LEFT", "M_RIGHT", self.taox_params, initial_conductance=0.25),
            "syn_R_to_L": TaOxSynapse("syn_R_to_L", "S_RIGHT", "M_LEFT", self.taox_params, initial_conductance=0.25),
            "syn_F_to_L": TaOxSynapse("syn_F_to_L", "S_FRONT", "M_LEFT", self.taox_params, initial_conductance=0.25),
            "syn_F_to_R": TaOxSynapse("syn_F_to_R", "S_FRONT", "M_RIGHT", self.taox_params, initial_conductance=0.25),
        }

        # Sliding Spike Windows for Motor Velocity Decoding (records spike timestamps in ms)
        self.motor_spike_windows: Dict[str, deque] = {
            m_id: deque() for m_id in self.motor_ids
        }

        # Recent spike tracking for STDP association (last spike time in ms)
        self.recent_pre_spikes: Dict[str, float] = {}
        self.recent_post_spikes: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset all neural and synaptic states."""
        self.sim_time_ms = 0.0
        for neuron in self.sensor_neurons.values():
            neuron.reset()
        for neuron in self.motor_neurons.values():
            neuron.reset()
        for synapse in self.synapses.values():
            synapse.reset()
        for window in self.motor_spike_windows.values():
            window.clear()
        for m_id in self.synaptic_currents:
            self.synaptic_currents[m_id] = 0.0
        self.recent_pre_spikes.clear()
        self.recent_post_spikes.clear()

    def encode_distance_to_rate(self, distance: float) -> float:
        """
        Convert obstacle distance (m) to spike firing rate (Hz):
            f(d) = f_max * (1 - min(d, d_max) / d_max)
        """
        clipped_dist = max(0.0, min(distance, self.config.d_max))
        freq = self.config.f_max * (1.0 - (clipped_dist / self.config.d_max))
        return max(0.0, freq)

    def _generate_poisson_spike(self, freq_hz: float, dt_ms: float) -> bool:
        """Evaluate whether a rate-coded spike occurs in interval dt."""
        if freq_hz <= 0.0:
            return False
        prob = (freq_hz * dt_ms) / 1000.0
        return random.random() < prob

    def step_ms(self, d_left: float, d_front: float, d_right: float) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """
        Execute one discrete simulation step of size dt (ms).
        
        Args:
            d_left (float): Distance reading to left obstacles (meters)
            d_front (float): Distance reading to front obstacles (meters)
            d_right (float): Distance reading to right obstacles (meters)
            
        Returns:
            Tuple[Dict[str, bool], Dict[str, bool]]: (pre_spikes, post_spikes)
        """
        import math
        self.sim_time_ms += self.config.dt
        dt = self.config.dt
        t_now = self.sim_time_ms

        # 1. SENSOR ENCODING: Calculate spike rates for each sensory sector
        rates = {
            "S_LEFT": self.encode_distance_to_rate(d_left),
            "S_FRONT": self.encode_distance_to_rate(d_front),
            "S_RIGHT": self.encode_distance_to_rate(d_right),
        }

        pre_spikes: Dict[str, bool] = {}
        for s_id, rate in rates.items():
            did_spike = self._generate_poisson_spike(rate, dt)
            pre_spikes[s_id] = did_spike
            if did_spike:
                self.sensor_neurons[s_id].force_spike(t_now)
                self.recent_pre_spikes[s_id] = t_now

        # 2. SYNAPTIC TRANSMISSION: Decay existing currents & accumulate incoming spikes
        decay_factor = math.exp(-dt / self.config.tau_syn)
        for m_id in self.motor_ids:
            self.synaptic_currents[m_id] *= decay_factor

        # Transmit pre-synaptic spikes through TaOx memristor synapses
        for syn_name, syn in self.synapses.items():
            if pre_spikes.get(syn.pre_id, False):
                injected_current = syn.compute_synaptic_current()
                self.synaptic_currents[syn.post_id] += injected_current

        # Compute total input current for each motor neuron
        total_motor_currents: Dict[str, float] = {}
        
        # Baseline forward drive: strong when front is clear, suppressed when blocked
        clearance_factor = max(0.0, min(1.0, d_front / self.config.d_max))
        fwd_current = self.config.baseline_forward_current * (clearance_factor ** 1.0)
        total_motor_currents["M_FWD"] = fwd_current

        # Exploratory tie-breaking when obstacle is straight ahead and side distances are symmetric
        exploratory_left = 0.0
        exploratory_right = 0.0
        if d_front < 0.8 and abs(d_left - d_right) < 0.25:
            # Randomly favor left or right to break symmetric trapping loops
            if random.random() < 0.5:
                exploratory_left = 12.0
            else:
                exploratory_right = 12.0

        total_motor_currents["M_LEFT"] = self.synaptic_currents["M_LEFT"] + exploratory_left
        total_motor_currents["M_RIGHT"] = self.synaptic_currents["M_RIGHT"] + exploratory_right

        # 3. POST-SYNAPTIC INTEGRATION & LIF SPIKING
        post_spikes: Dict[str, bool] = {}
        for m_id, neuron in self.motor_neurons.items():
            did_spike = neuron.step(total_motor_currents[m_id], dt, t_now)
            post_spikes[m_id] = did_spike
            if did_spike:
                self.motor_spike_windows[m_id].append(t_now)
                self.recent_post_spikes[m_id] = t_now

        # 4. MEMRISTIVE STDP ONLINE LEARNING
        # Update TaOx conductances based on pre/post spike event timing
        self._apply_stdp_learning(pre_spikes, post_spikes, t_now)

        return pre_spikes, post_spikes

    def _apply_stdp_learning(
        self,
        pre_spikes: Dict[str, bool],
        post_spikes: Dict[str, bool],
        t_now: float
    ) -> None:
        """
        Apply STDP updates to TaOx synapses when pre or post spikes occur.
        """
        # Temporal STDP pairing window limit (ms)
        stdp_cutoff = 50.0

        for syn in self.synapses.values():
            # Post-synaptic spike occurred: Check recent pre-synaptic spike (LTP: t_post > t_pre)
            if post_spikes.get(syn.post_id, False):
                t_pre = self.recent_pre_spikes.get(syn.pre_id, None)
                if t_pre is not None and (t_now - t_pre) <= stdp_cutoff:
                    syn.update_stdp(t_pre=t_pre, t_post=t_now, current_time=t_now)

            # Pre-synaptic spike occurred: Check recent post-synaptic spike (LTD: t_post < t_pre)
            if pre_spikes.get(syn.pre_id, False):
                t_post = self.recent_post_spikes.get(syn.post_id, None)
                if t_post is not None and (t_now - t_post) <= stdp_cutoff:
                    syn.update_stdp(t_pre=t_now, t_post=t_post, current_time=t_now)

    def decode_motor_commands(self) -> MotorCommand:
        """
        Decode accumulated motor spikes into robot linear and angular velocity commands.
            v = kv * (Count_forward / ΔT)
            omega = k_omega * ((Count_left - Count_right) / ΔT)
        """
        t_now = self.sim_time_ms
        window_cutoff = t_now - self.config.window_dt

        # Prune old spikes outside evaluation window ΔT
        counts: Dict[str, int] = {}
        for m_id, window in self.motor_spike_windows.items():
            while window and window[0] < window_cutoff:
                window.popleft()
            counts[m_id] = len(window)

        # Countforward / ΔT
        dt_sec = self.config.window_dt / 1000.0
        count_fwd = counts["M_FWD"]
        count_left = counts["M_LEFT"]
        count_right = counts["M_RIGHT"]

        # Linear Velocity (v)
        v = self.config.k_v * (count_fwd / dt_sec)
        v = max(0.0, min(self.config.max_linear_vel, v))

        # Angular Velocity (omega)
        # Countleft > Countright produces positive angular velocity (turn left)
        omega = self.config.k_omega * ((count_left - count_right) / dt_sec)
        omega = max(-self.config.max_angular_vel, min(self.config.max_angular_vel, omega))

        # Collision avoidance reflex: If turning sharply, slow down linear speed
        if abs(omega) > 0.5:
            v *= max(0.1, 1.0 - (abs(omega) / self.config.max_angular_vel) * 0.8)

        return MotorCommand(linear_x=v, angular_z=omega)

    def process_sensor_cycle(
        self,
        d_left: float,
        d_front: float,
        d_right: float,
        duration_ms: float = 50.0
    ) -> MotorCommand:
        """
        Process incoming sensor readings over a discrete control cycle (e.g. 50 ms = 20 Hz ROS loop).
        Steps internal LIF & TaOx differential equations with dt = 1.0 ms.
        
        Returns:
            MotorCommand: Decoded linear and angular velocities
        """
        num_steps = int(duration_ms / self.config.dt)
        for _ in range(num_steps):
            self.step_ms(d_left, d_front, d_right)

        return self.decode_motor_commands()
