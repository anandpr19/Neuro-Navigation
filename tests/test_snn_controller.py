"""
Unit tests for NeuromorphicRobotController
Validates sensor rate encoding, motor decoding, and closed-loop control dynamics.
"""

import pytest
from core.snn_controller import NeuromorphicRobotController, SNNConfig


def test_sensor_rate_encoding():
    """Verify rate encoding formula: f(d) = f_max * (1 - min(d, d_max) / d_max)."""
    config = SNNConfig(f_max=100.0, d_max=2.0)
    controller = NeuromorphicRobotController(config=config)
    
    # At d = 0, rate should be f_max
    assert controller.encode_distance_to_rate(0.0) == 100.0
    
    # At d = d_max, rate should be 0.0
    assert controller.encode_distance_to_rate(2.0) == 0.0
    
    # At d > d_max, rate should be 0.0
    assert controller.encode_distance_to_rate(3.5) == 0.0
    
    # At d = d_max / 2, rate should be f_max / 2
    assert controller.encode_distance_to_rate(1.0) == 50.0


def test_motor_velocity_decoding_forward():
    """Verify linear and angular velocity decoding when forward motor spikes."""
    config = SNNConfig(k_v=0.015, k_omega=0.12, window_dt=100.0)
    controller = NeuromorphicRobotController(config=config)
    controller.sim_time_ms = 100.0
    
    # Add spikes to forward window
    controller.motor_spike_windows["M_FWD"].append(90.0)
    controller.motor_spike_windows["M_FWD"].append(95.0)
    
    cmd = controller.decode_motor_commands()
    # v = k_v * (Count / 0.1) = 0.015 * (2 / 0.1) = 0.3 -> capped at max_linear_vel
    assert cmd.linear_x == min(config.max_linear_vel, 0.3)
    assert cmd.angular_z == 0.0


def test_motor_velocity_decoding_turns():
    """Verify angular velocity turns left when M_LEFT spikes more than M_RIGHT."""
    config = SNNConfig(k_omega=0.12, window_dt=100.0)
    controller = NeuromorphicRobotController(config=config)
    controller.sim_time_ms = 100.0
    
    controller.motor_spike_windows["M_LEFT"].append(90.0)
    cmd = controller.decode_motor_commands()
    assert cmd.angular_z > 0.0  # Turn left
    
    # When M_RIGHT has more spikes, turn right
    controller.motor_spike_windows["M_LEFT"].clear()
    controller.motor_spike_windows["M_RIGHT"].append(90.0)
    cmd_right = controller.decode_motor_commands()
    assert cmd_right.angular_z < 0.0  # Turn right


def test_obstacle_avoidance_turn_response():
    """Verify presenting obstacle on left triggers avoidance turn towards right."""
    controller = NeuromorphicRobotController()
    
    # Present obstacle on left (d_left = 0.2m, d_front = 2.0m, d_right = 2.0m)
    # Step for 500 ms to accumulate spikes
    for _ in range(500):
        controller.step_ms(d_left=0.2, d_front=2.0, d_right=2.0)
        
    cmd = controller.decode_motor_commands()
    # Evasive turn away from left obstacle -> omega < 0 (turn right)
    assert cmd.angular_z < 0.0
