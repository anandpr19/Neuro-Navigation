"""
Neuromorphic ROS 2 Brain Controller Node
Subscribes to /scan (LaserScan), executes LIF SNN and TaOx Memristor STDP,
publishes to /cmd_vel (both Twist and TwistStamped for ROS 2 Jazzy compatibility).
Includes live telemetry logging and automatic output visualization.
"""

import sys
import os
import math
from typing import List

# Ensure core module is accessible when running as a node
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist, TwistStamped
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object  # Fallback type

from core.snn_controller import NeuromorphicRobotController, SNNConfig
from core.lif_neuron import LIFParameters
from core.taox_synapse import TaOxParameters
from core.telemetry import TelemetryLogger


class NeuromorphicBrainNode(Node):
    """
    ROS 2 Node encapsulating the Neuromorphic SNN and TaOx STDP Controller.
    """

    def __init__(self):
        super().__init__("neuromorphic_brain_node")
        self.get_logger().info("Initializing Neuromorphic Brain Controller Node...")

        # Declare ROS parameters
        self.declare_parameter("f_max", 100.0)
        self.declare_parameter("d_max", 1.20)
        self.declare_parameter("max_linear_vel", 0.22)
        self.declare_parameter("max_angular_vel", 1.50)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("output_dir", "simulation_results")

        # Retrieve parameters
        f_max = self.get_parameter("f_max").get_parameter_value().double_value
        d_max = self.get_parameter("d_max").get_parameter_value().double_value
        max_v = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        max_omega = self.get_parameter("max_angular_vel").get_parameter_value().double_value
        self.control_rate_hz = self.get_parameter("control_rate_hz").get_parameter_value().double_value
        self.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize SNN Controller & Telemetry Logger
        config = SNNConfig(
            f_max=f_max,
            d_max=d_max,
            max_linear_vel=max_v,
            max_angular_vel=max_omega
        )
        self.controller = NeuromorphicRobotController(config=config)
        self.telemetry = TelemetryLogger(self.controller)

        # ROS 2 Pub/Sub
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )
        # Publish Twist (Standard ROS 2)
        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )
        # Publish TwistStamped (ROS 2 Jazzy TurtleBot3 default bridge requirement)
        self.cmd_stamped_pub = self.create_publisher(
            TwistStamped,
            "/cmd_vel_stamped",
            10
        )

        # State cache for latest sector distances
        self.d_left: float = d_max
        self.d_front: float = d_max
        self.d_right: float = d_max
        self.scan_count: int = 0

        # Control Loop Timer (20 Hz)
        timer_period = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)
        self.get_logger().info(f"Neuromorphic Brain Node Running at {self.control_rate_hz} Hz")

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Process incoming 360-degree LiDAR LaserScan message.
        Partitions ranges into Front, Left, Right sectors.
        """
        if not msg.ranges:
            return

        self.scan_count += 1
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        front_ranges: List[float] = []
        left_ranges: List[float] = []
        right_ranges: List[float] = []

        max_range = self.controller.config.d_max

        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r) or r <= msg.range_min or r >= msg.range_max:
                continue

            angle_rad = angle_min + (i * angle_increment)
            angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
            angle_deg = math.degrees(angle_rad)

            # Front Sector (-15° to +15°)
            if -15.0 <= angle_deg <= 15.0:
                front_ranges.append(r)
            # Left Sector (+15° to +45°)
            elif 15.0 < angle_deg <= 45.0:
                left_ranges.append(r)
            # Right Sector (-45° to -15°)
            elif -45.0 <= angle_deg < -15.0:
                right_ranges.append(r)

        self.d_front = min(front_ranges) if front_ranges else max_range
        self.d_left = min(left_ranges) if left_ranges else max_range
        self.d_right = min(right_ranges) if right_ranges else max_range

    def control_loop_callback(self) -> None:
        """
        Executes one control cycle of duration (1000 / rate) ms.
        Integrates LIF & TaOx equations, decodes spikes, and publishes Twist velocity.
        """
        cycle_duration_ms = 1000.0 / self.control_rate_hz

        # Step internal SNN equations over cycle
        motor_cmd = self.controller.process_sensor_cycle(
            d_left=self.d_left,
            d_front=self.d_front,
            d_right=self.d_right,
            duration_ms=cycle_duration_ms
        )

        # 1. Publish geometry_msgs/msg/Twist command
        twist = Twist()
        twist.linear.x = float(motor_cmd.linear_x)
        twist.angular.z = float(motor_cmd.angular_z)
        self.cmd_pub.publish(twist)

        # 2. Publish geometry_msgs/msg/TwistStamped command for ROS 2 Jazzy Gazebo bridge
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = "base_footprint"
        twist_stamped.twist.linear.x = float(motor_cmd.linear_x)
        twist_stamped.twist.angular.z = float(motor_cmd.angular_z)
        self.cmd_stamped_pub.publish(twist_stamped)

        # Record step into live telemetry logger
        self.telemetry.record_step(
            t_ms=self.controller.sim_time_ms,
            pre_spikes={},
            post_spikes={},
            v_lin=motor_cmd.linear_x,
            omega=motor_cmd.angular_z,
            d_left=self.d_left,
            d_front=self.d_front,
            d_right=self.d_right
        )

        # Periodic console logging and telemetry plot saving (every 5 seconds)
        sim_time_sec = self.controller.sim_time_ms / 1000.0
        if int(self.controller.sim_time_ms) % 5000 < int(cycle_duration_ms):
            conductances = [round(s.g, 3) for s in self.controller.synapses.values()]
            self.get_logger().info(
                f"[{sim_time_sec:5.1f}s] Scans={self.scan_count} | "
                f"Distances: L={self.d_left:.2f}m F={self.d_front:.2f}m R={self.d_right:.2f}m | "
                f"Cmd: v={motor_cmd.linear_x:.2f}m/s ω={motor_cmd.angular_z:.2f}rad/s | "
                f"Conductances: {conductances}"
            )
            plot_path = os.path.join(self.output_dir, "ros2_telemetry_live.png")
            self.telemetry.save_plot(plot_path)


def main(args=None):
    if not HAS_ROS2:
        print("[ERROR] rclpy is not installed in current environment.")
        sys.exit(1)

    rclpy.init(args=args)
    node = NeuromorphicBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Neuromorphic Brain Controller Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
