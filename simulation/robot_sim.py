"""
Kinematic Robot Model with Raycasting LiDAR Sensor
Simulates differential drive motion and laser scans identical to TurtleBot3 in ROS/Gazebo.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .arena import Arena


@dataclass
class SectorReadings:
    """Minimum distance readings across the three biological sensory sectors."""
    d_left: float    # Left sector (approx +20 to +70 deg)
    d_front: float   # Front sector (approx -20 to +20 deg)
    d_right: float   # Right sector (approx -70 to -20 deg)


class SimulatedRobot:
    """
    Differential drive robot (TurtleBot3 equivalent) executing in the 2D arena.
    """

    def __init__(
        self,
        x: float = 2.0,
        y: float = 1.5,
        theta: float = 0.0,
        max_lidar_range: float = 3.5,
        radius: float = 0.15
    ):
        self.x = x
        self.y = y
        self.theta = theta  # Heading in radians
        self.max_lidar_range = max_lidar_range
        self.radius = radius
        self.trajectory: List[Tuple[float, float]] = [(x, y)]

    def step(self, v_linear: float, omega_angular: float, dt_sec: float) -> None:
        """Update robot pose using unicycle kinematics."""
        self.x += v_linear * math.cos(self.theta) * dt_sec
        self.y += v_linear * math.sin(self.theta) * dt_sec
        self.theta = (self.theta + omega_angular * dt_sec + math.pi) % (2.0 * math.pi) - math.pi
        self.trajectory.append((self.x, self.y))

    def scan_lidar(self, arena: Arena, num_rays_per_sector: int = 7) -> SectorReadings:
        """
        Cast simulated LiDAR rays across three forward sectors:
        - Front: [-20°, +20°] relative to heading
        - Left:  [+20°, +70°] relative to heading
        - Right: [-70°, -20°] relative to heading
        
        Returns the minimum distance reading in each sector.
        """
        front_dists: List[float] = []
        left_dists: List[float] = []
        right_dists: List[float] = []

        # Front Sector (-20° to +20°)
        for i in range(num_rays_per_sector):
            angle_offset = math.radians(-20.0 + (40.0 * i / max(1, num_rays_per_sector - 1)))
            ray_angle = self.theta + angle_offset
            dist = arena.cast_ray(self.x, self.y, ray_angle, self.max_lidar_range)
            front_dists.append(dist)

        # Left Sector (+20° to +70°)
        for i in range(num_rays_per_sector):
            angle_offset = math.radians(20.0 + (50.0 * i / max(1, num_rays_per_sector - 1)))
            ray_angle = self.theta + angle_offset
            dist = arena.cast_ray(self.x, self.y, ray_angle, self.max_lidar_range)
            left_dists.append(dist)

        # Right Sector (-70° to -20°)
        for i in range(num_rays_per_sector):
            angle_offset = math.radians(-70.0 + (50.0 * i / max(1, num_rays_per_sector - 1)))
            ray_angle = self.theta + angle_offset
            dist = arena.cast_ray(self.x, self.y, ray_angle, self.max_lidar_range)
            right_dists.append(dist)

        return SectorReadings(
            d_left=min(left_dists) if left_dists else self.max_lidar_range,
            d_front=min(front_dists) if front_dists else self.max_lidar_range,
            d_right=min(right_dists) if right_dists else self.max_lidar_range,
        )
