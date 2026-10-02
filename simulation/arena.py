"""
2D Arena Simulation Environment with Obstacles and Raycasting
Provides a 2D physical world mimicking a Gazebo maze for autonomous navigation testing.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LineSegment:
    """Represents a wall or obstacle boundary line from (x1, y1) to (x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    def intersect_ray(self, ox: float, oy: float, angle_rad: float, max_range: float) -> Optional[float]:
        """
        Calculates distance from ray origin (ox, oy) along angle_rad to intersection.
        Returns distance if hit occurs within max_range, else None.
        """
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        x3, y3 = ox, oy
        x4, y4 = ox + dx * max_range, oy + dy * max_range

        x1, y1 = self.x1, self.y1
        x2, y2 = self.x2, self.y2

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-9:
            return None  # Parallel

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
            hit_x = x1 + ua * (x2 - x1)
            hit_y = y1 + ua * (y2 - y1)
            dist = math.hypot(hit_x - ox, hit_y - oy)
            return dist

        return None


class Arena:
    """Bounded 2D arena with perimeter walls and interior obstacle blocks."""

    def __init__(self, width: float = 10.0, height: float = 10.0):
        self.width = width
        self.height = height
        self.walls: List[LineSegment] = []
        self._build_default_maze()

    def add_box(self, cx: float, cy: float, w: float, h: float) -> None:
        """Add rectangular obstacle centered at (cx, cy) with dimensions (w, h)."""
        x_min, x_max = cx - w / 2.0, cx + w / 2.0
        y_min, y_max = cy - h / 2.0, cy + h / 2.0
        self.walls.append(LineSegment(x_min, y_min, x_max, y_min))
        self.walls.append(LineSegment(x_max, y_min, x_max, y_max))
        self.walls.append(LineSegment(x_max, y_max, x_min, y_max))
        self.walls.append(LineSegment(x_min, y_max, x_min, y_min))

    def _build_default_maze(self) -> None:
        """Instantiate perimeter walls and challenging interior obstacles."""
        W, H = self.width, self.height
        # Outer boundary
        self.walls.append(LineSegment(0, 0, W, 0))
        self.walls.append(LineSegment(W, 0, W, H))
        self.walls.append(LineSegment(W, H, 0, H))
        self.walls.append(LineSegment(0, H, 0, 0))

        # Interior pillars and barriers (maze layout)
        self.add_box(3.0, 3.0, 1.2, 1.2)
        self.add_box(7.0, 3.0, 1.2, 1.2)
        self.add_box(5.0, 5.5, 2.5, 0.8)
        self.add_box(3.0, 7.5, 1.0, 1.5)
        self.add_box(7.0, 7.5, 1.0, 1.5)

    def cast_ray(self, ox: float, oy: float, angle_rad: float, max_range: float) -> float:
        """Return the closest obstacle distance along a ray up to max_range."""
        min_dist = max_range
        for wall in self.walls:
            d = wall.intersect_ray(ox, oy, angle_rad, max_range)
            if d is not None and d < min_dist:
                min_dist = d
        return min_dist
