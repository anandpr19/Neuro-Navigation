"""
Leaky Integrate-and-Fire (LIF) Neuron Model
Governed by differential equation:
    tau_m * (dV(t) / dt) = -(V(t) - V_rest) + R_m * I(t)
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LIFParameters:
    """Nominal parameters for biological LIF neuron from technical specification."""
    v_rest: float = -70.0      # Resting membrane potential (mV)
    v_th: float = -55.0        # Threshold potential to emit action potential (mV)
    v_reset: float = -70.0     # Reset potential after spike (mV)
    tau_m: float = 20.0        # Membrane time constant (ms)
    r_m: float = 1.0           # Membrane resistance (MOhm)
    tau_ref: float = 2.0       # Refractory period duration (ms)


class LIFNeuron:
    """
    Simulation of a biological Leaky Integrate-and-Fire (LIF) neuron.
    
    Attributes:
        neuron_id (str): Identifier for neuron (e.g., 'S_LEFT', 'M_FWD')
        params (LIFParameters): Biophysical neuron parameters
        v (float): Current membrane potential V(t) in mV
        refractory_time_left (float): Remaining refractory duration in ms
        last_spike_time (Optional[float]): Timestamp (ms) of the most recent spike
        spike_times (List[float]): Complete history of fired action potential timestamps (ms)
    """

    def __init__(self, neuron_id: str, params: Optional[LIFParameters] = None):
        self.neuron_id = neuron_id
        self.params = params or LIFParameters()
        self.v: float = self.params.v_rest
        self.refractory_time_left: float = 0.0
        self.last_spike_time: Optional[float] = None
        self.spike_times: List[float] = []

    def reset(self) -> None:
        """Reset membrane state to baseline resting conditions."""
        self.v = self.params.v_rest
        self.refractory_time_left = 0.0
        self.last_spike_time = None
        self.spike_times.clear()

    def step(self, current_i: float, dt: float, current_time: float) -> bool:
        """
        Integrate the neuron state over a time step dt (ms).
        
        Args:
            current_i (float): Total input synaptic current I(t) (nA)
            dt (float): Integration step size in milliseconds (e.g. 1.0 ms)
            current_time (float): Current simulation clock timestamp (ms)
            
        Returns:
            bool: True if an action potential (spike) was emitted, False otherwise.
        """
        # If in refractory period, clamp voltage to reset potential
        if self.refractory_time_left > 0.0:
            self.refractory_time_left = max(0.0, self.refractory_time_left - dt)
            self.v = self.params.v_reset
            return False

        # Euler numerical integration of LIF differential equation:
        # dV/dt = [ -(V(t) - V_rest) + R_m * I(t) ] / tau_m
        dv_dt = (-(self.v - self.params.v_rest) + (self.params.r_m * current_i)) / self.params.tau_m
        self.v += dv_dt * dt

        # Spike generation test
        if self.v >= self.params.v_th:
            self.v = self.params.v_reset
            self.refractory_time_left = self.params.tau_ref
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
            return True

        return False

    def force_spike(self, current_time: float) -> None:
        """Manually trigger a spike (used for sensory rate encoding injection)."""
        self.v = self.params.v_reset
        self.refractory_time_left = self.params.tau_ref
        self.last_spike_time = current_time
        self.spike_times.append(current_time)
