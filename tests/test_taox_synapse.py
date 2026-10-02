"""
Unit tests for TaOx Memristor Synapse Model with STDP
Validates mathematical equations for LTP, LTD, and conductance bounds.
"""

import pytest
import math
from core.taox_synapse import TaOxSynapse, TaOxParameters


def test_taox_initial_conductance():
    """Verify initial conductance initialization and clamping."""
    params = TaOxParameters(g_min=0.001, g_max=1.0, initial_g=0.2)
    syn = TaOxSynapse("test_syn", "pre", "post", params)
    assert syn.g == 0.2
    assert len(syn.conductance_history) == 1


def test_taox_ltp_stdp():
    """Verify Long-Term Potentiation (LTP) when delta_t = t_post - t_pre > 0."""
    params = TaOxParameters(a_plus=0.01, tau_plus=15.0, g_min=0.001, g_max=1.0, initial_g=0.2)
    syn = TaOxSynapse("test_syn", "pre", "post", params)
    
    t_pre = 10.0
    t_post = 25.0
    delta_t = t_post - t_pre  # 15.0 ms = 1 * tau_plus
    
    delta_g = syn.update_stdp(t_pre=t_pre, t_post=t_post, current_time=25.0)
    
    # Theoretical delta_g = A+ * exp(-delta_t / tau+)
    expected_delta_g = params.a_plus * math.exp(-15.0 / params.tau_plus)
    assert pytest.approx(delta_g, rel=1e-4) == expected_delta_g
    assert pytest.approx(syn.g, rel=1e-4) == 0.2 + expected_delta_g


def test_taox_ltd_stdp():
    """Verify Long-Term Depression (LTD) when delta_t = t_post - t_pre < 0."""
    params = TaOxParameters(a_minus=0.01, tau_minus=15.0, g_min=0.001, g_max=1.0, initial_g=0.2)
    syn = TaOxSynapse("test_syn", "pre", "post", params)
    
    t_pre = 30.0
    t_post = 15.0
    delta_t = t_post - t_pre  # -15.0 ms = -1 * tau_minus
    
    delta_g = syn.update_stdp(t_pre=t_pre, t_post=t_post, current_time=30.0)
    
    # Theoretical delta_g = -A- * exp(delta_t / tau-)
    expected_delta_g = -params.a_minus * math.exp(-15.0 / params.tau_minus)
    assert pytest.approx(delta_g, rel=1e-4) == expected_delta_g
    assert pytest.approx(syn.g, rel=1e-4) == 0.2 + expected_delta_g


def test_taox_conductance_clipping_bounds():
    """Verify conductance G cannot exceed G_max or fall below G_min."""
    params = TaOxParameters(g_min=0.001, g_max=1.0, a_plus=0.5, a_minus=0.5, initial_g=0.9)
    syn = TaOxSynapse("test_syn", "pre", "post", params)
    
    # Push past G_max
    syn.update_stdp(t_pre=10.0, t_post=11.0, current_time=11.0)
    assert syn.g <= params.g_max
    assert syn.g == 1.0
    
    # Push below G_min
    for _ in range(25):
        syn.update_stdp(t_pre=11.0, t_post=10.0, current_time=30.0)
    assert syn.g >= params.g_min
    assert syn.g == params.g_min
