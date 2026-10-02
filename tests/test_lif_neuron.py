"""
Unit tests for Leaky Integrate-and-Fire (LIF) Neuron Model
Validates mathematical equations and biophysical dynamics from technical specification.
"""

import pytest
import math
from core.lif_neuron import LIFNeuron, LIFParameters


def test_lif_initial_state():
    """Verify initial resting conditions."""
    params = LIFParameters()
    neuron = LIFNeuron("test_n", params)
    assert neuron.v == params.v_rest
    assert neuron.refractory_time_left == 0.0
    assert neuron.last_spike_time is None
    assert len(neuron.spike_times) == 0


def test_lif_passive_membrane_leak():
    """Verify exponential voltage leakage towards V_rest in absence of input current."""
    params = LIFParameters(v_rest=-70.0, tau_m=20.0, r_m=1.0)
    neuron = LIFNeuron("test_n", params)
    
    # Displace membrane voltage to -60 mV
    neuron.v = -60.0
    dt = 1.0  # ms
    
    # Step with 0 current for 20 ms (1 time constant tau_m)
    for i in range(20):
        neuron.step(current_i=0.0, dt=dt, current_time=i * dt)
    
    # Theoretical decay: V(t) - V_rest = (V_0 - V_rest) * exp(-t / tau_m)
    expected_v = params.v_rest + (-60.0 - params.v_rest) * math.exp(-20.0 / params.tau_m)
    assert pytest.approx(neuron.v, abs=0.5) == expected_v


def test_lif_spike_generation_and_reset():
    """Verify neuron emits spike when V >= V_th and resets to V_reset."""
    params = LIFParameters(v_rest=-70.0, v_th=-55.0, v_reset=-70.0, tau_m=20.0, r_m=1.0)
    neuron = LIFNeuron("test_n", params)
    
    # Inject strong current to trigger spike
    spiked = False
    current_time = 0.0
    dt = 1.0
    while current_time < 50.0 and not spiked:
        current_time += dt
        spiked = neuron.step(current_i=30.0, dt=dt, current_time=current_time)
        
    assert spiked is True
    assert neuron.v == params.v_reset
    assert neuron.refractory_time_left == params.tau_ref
    assert neuron.last_spike_time == current_time
    assert len(neuron.spike_times) == 1


def test_lif_refractory_period_clamping():
    """Verify membrane potential is clamped during refractory period."""
    params = LIFParameters(v_rest=-70.0, v_th=-55.0, v_reset=-70.0, tau_ref=2.0)
    neuron = LIFNeuron("test_n", params)
    
    # Trigger spike
    neuron.force_spike(current_time=10.0)
    assert neuron.refractory_time_left == 2.0
    
    # Step during refractory period with massive input current
    spiked = neuron.step(current_i=1000.0, dt=1.0, current_time=11.0)
    assert spiked is False
    assert neuron.v == params.v_reset
    assert neuron.refractory_time_left == 1.0
    
    # After refractory expires, current can depolarize neuron again
    neuron.step(current_i=1000.0, dt=1.0, current_time=12.0)
    assert neuron.refractory_time_left == 0.0
