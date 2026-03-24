"""
Script to visualize individual DOF movements of a robot.
Each joint moves one at a time through its full range.
"""

import numpy as np
import sys
from pathlib import Path
import time
from typing import Optional

from fire import Fire
import vamp

sys.path.insert(0, str(Path(__file__).parent.parent))
from viser_visualizer import ViserVisualizer


def interpolate_path_adaptive(simplified_path):
    """
    Interpolate path with distance-based adaptive interpolation for smooth animation.
    
    Args:
        simplified_path: Array of states to interpolate
        
    Returns:
        Interpolated path as numpy array
    """
    interp_path = []
    
    # Calculate distances between consecutive states
    distances = []
    for j in range(len(simplified_path) - 1):
        dist = np.linalg.norm(simplified_path[j + 1] - simplified_path[j])
        distances.append(dist)
    
    # Normalize distances and map to interpolation points
    max_distance = max(distances) if distances else 1.0
    min_points = 10  # minimum interpolation points per segment
    max_points = 12  # maximum interpolation points per segment
    
    for j in range(len(simplified_path) - 1):
        # Scale number of points based on relative distance
        relative_distance = distances[j] / max_distance if max_distance > 0 else 1.0
        num_points = max(min_points, int(min_points + relative_distance * (max_points - min_points)))
        
        segment = np.linspace(simplified_path[j], simplified_path[j + 1], num=num_points, endpoint=False)
        interp_path.extend(segment)
    
    # Don't forget the final state
    interp_path.append(simplified_path[-1])
    return np.array(interp_path)


def main(
    robot: str = "panda",  # Robot to visualize
    port: Optional[int] = None,      # Optional port for viser server
    fps: float = 30.0,     # Frames per second for animation
    start_config: Optional[str] = None,  # Starting configuration as comma-separated values
):
    """
    Visualize individual DOF movements of a robot in a single trajectory.
    Each DOF moves sequentially a little bit from its starting position.
    
    Args:
        robot: Robot name (e.g., 'panda', 'ur5', 'fetch')
        port: Optional port for viser server
        fps: Animation frame rate
        start_config: Starting configuration as comma-separated values (e.g., "0.1,1.32,1.4,-0.2,1.72,0,1.66,0")
    """
    
    # Try to get robot module
    try:
        robot_module = getattr(vamp, robot)
    except AttributeError:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")
    
    # Get robot bounds
    upper_bounds = robot_module.upper_bounds()
    lower_bounds = robot_module.lower_bounds()
    dimension = robot_module.dimension()
    joint_names = robot_module.joint_names()
    
    print(f"Robot: {robot}")
    print(f"Dimension: {dimension}")
    print(f"Joint names: {joint_names}")
    print(f"\nJoint bounds:")
    for i, (name, lower, upper) in enumerate(zip(joint_names, lower_bounds, upper_bounds)):
        print(f"  [{i}] {name}: [{lower:.4f}, {upper:.4f}]")
    
    # Parse goal configuration (the parameter is used as goal, not start)
    if start_config is not None:
        try:
            config_values = [float(x.strip()) for x in start_config.split(',')]
            if len(config_values) != dimension:
                raise ValueError(f"Expected {dimension} values, got {len(config_values)}")
            goal_config = np.array(config_values)
            print(f"\nUsing provided goal config: {goal_config}")
        except Exception as e:
            raise ValueError(f"Invalid start_config format: {e}")
    else:
        # Default goal to middle of range
        goal_config = (lower_bounds + upper_bounds) / 2.0
        print(f"\nUsing default goal config (middle of range): {goal_config}")
    
    # Initialize visualizer
    vis = ViserVisualizer(robot_name=robot, port=port)
    
    # Set camera to look at the robot
    vis.set_camera(position=[1.5, 1.5, 1.5], target=[0, 0, 1.0])
    
    # Frame timing
    frame_time = 1.0 / fps
    
    # Start position is more retracted (lower joint angles, 25% of range)
    trajectory_start = [0.0, 1.0, 1.0, 2.5, -2.0, 2.5, 2.0, 1.5]
    
    print(f"Start config (retracted): {trajectory_start}")
    print(f"Goal config: {goal_config}")
    
    # Build trajectory where each DOF moves sequentially from start toward goal
    trajectory_points = [trajectory_start.copy()]
    
    # Number of intermediate points per DOF for more movement
    steps_per_dof = 5  # More fine-grained movement
    print(f"dimension: {dimension}")
    for dof in range(dimension):
        current_config = trajectory_points[-1].copy()
        
        # Interpolate this DOF from current to goal value
        start_val = current_config[dof]
        goal_val = goal_config[dof]
        
        # Create multiple intermediate steps for this DOF
        for step in range(1, steps_per_dof + 1):
            fraction = step / steps_per_dof
            new_config = current_config.copy()
            new_config[dof] = start_val + (goal_val - start_val) * fraction
            
            # Clamp to bounds (should already be in bounds, but just in case)
            new_config[dof] = np.clip(new_config[dof], lower_bounds[dof], upper_bounds[dof])
            
            trajectory_points.append(new_config)
        
        print(f"DOF {dof} ({joint_names[dof]}): {start_val:.4f} -> {goal_val:.4f}")
    
    # Convert to numpy array and interpolate
    trajectory_points = np.array(trajectory_points)
    trajectory = interpolate_path_adaptive(trajectory_points)
    
    print(f"\nTotal trajectory points (after interpolation): {len(trajectory)}")
    
    # Visualize trajectory
    vis.reset()
    vis.set_camera(position=[1.0, 1.0, 1.5], target=[0, 0, 0.4])
    vis.visualize_trajectory(trajectory)
    
    # Play the animation
    print(f"\nAnimating all DOFs sequentially...")
    print("Press any key to stop the animation")
    
    key = vis.play_until_key_pressed(key='any', dt=frame_time)
    
    print("\nDone! Closed visualizer.")


if __name__ == "__main__":
    Fire(main)
