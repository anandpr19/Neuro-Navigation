"""
Tantalum Oxide (TaOx) Memristive Synapse Model
Emulates nanoscale resistive switching and Spike-Timing-Dependent Plasticity (STDP):
    ΔG = A+ * exp(-Δt / tau+)   for Δt > 0 (LTP)
    ΔG = -A- * exp(Δt / tau-)   for Δt < 0 (LTD)
    G = clip(G + ΔG, Gmin, Gmax)
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class TaOxParameters:
    """Nominal parameters for TaOx memristor dynamics from technical specification."""
    a_plus: float = 0.01        # LTP learning rate amplitude (Siemens)
    a_minus: float = 0.01       # LTD learning rate amplitude (Siemens)
    tau_plus: float = 15.0      # LTP temporal window constant (ms)
    tau_minus: float = 15.0     # LTD temporal window constant (ms)
    g_min: float = 0.001        # Minimum device conductance (S)
    g_max: float = 1.0          # Maximum device conductance (S)
    initial_g: float = 0.15     # Initial conductance weight (S)
    synaptic_weight_scale: float = 150.0  # Current injection scaling factor (nA / S)


class TaOxSynapse:
    """
    Simulation of a synthetic synapse implemented with a simulated TaOx memristor.
    
    Attributes:
        synapse_id (str): Identifier connecting pre_id -> post_id
        pre_id (str): Pre-synaptic neuron identifier
        post_id (str): Post-synaptic neuron identifier
        params (TaOxParameters): Physical memristor parameters
        g (float): Current memristive conductance G (S)
        conductance_history (List[Tuple[float, float]]): (timestamp_ms, conductance_S)
    """

    def __init__(
        self,
        synapse_id: str,
        pre_id: str,
        post_id: str,
        params: Optional[TaOxParameters] = None,
        initial_conductance: Optional[float] = None
    ):
        self.synapse_id = synapse_id
        self.pre_id = pre_id
        self.post_id = post_id
        self.params = params or TaOxParameters()
        
        # Initialize conductance within bounds
        init_g = initial_conductance if initial_conductance is not None else self.params.initial_g
        self.g: float = self._clip_conductance(init_g)
        self.conductance_history: List[Tuple[float, float]] = [(0.0, self.g)]

    def _clip_conductance(self, value: float) -> float:
        """Clamp conductance to physical device limits."""
        return max(self.params.g_min, min(self.params.g_max, value))

    def reset(self, initial_conductance: Optional[float] = None) -> None:
        """Reset conductance to initial state."""
        init_g = initial_conductance if initial_conductance is not None else self.params.initial_g
        self.g = self._clip_conductance(init_g)
        self.conductance_history = [(0.0, self.g)]

    def update_stdp(self, t_pre: float, t_post: float, current_time: float) -> float:
        """
        Compute and apply STDP weight update based on spike timing delta:
            delta_t = t_post - t_pre
            
        Args:
            t_pre (float): Timestamp of pre-synaptic spike (ms)
            t_post (float): Timestamp of post-synaptic spike (ms)
            current_time (float): Current simulation time (ms) for logging
            
        Returns:
            float: Conductance delta (ΔG) applied
        """
        delta_t = t_post - t_pre

        if delta_t > 0.0:
            # Long-Term Potentiation (LTP): pre fired before post
            delta_g = self.params.a_plus * math.exp(-delta_t / self.params.tau_plus)
        elif delta_t < 0.0:
            # Long-Term Depression (LTD): pre fired after post
            delta_g = -self.params.a_minus * math.exp(delta_t / self.params.tau_minus)
        else:
            delta_g = 0.0

        old_g = self.g
        self.g = self._clip_conductance(self.g + delta_g)
        actual_delta = self.g - old_g
        self.conductance_history.append((current_time, self.g))
        return actual_delta

    def compute_synaptic_current(self) -> float:
        """
        Return the synaptic current injected into post-synaptic neuron
        when a pre-synaptic spike arrives, scaled by conductance G.
        """
        return self.g * self.params.synaptic_weight_scale
