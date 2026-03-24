import pickle
import time
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Union, List
from functools import partial
import sys
import subprocess
import numpy as np

from fire import Fire
import vamp
from vamp import pointcloud as vpc
from ompl import base as ob
from ompl import geometric as og
from validity import VampMotionValidator, isStateValid

sys.path.insert(0, str(Path(__file__).parent.parent))
from viser_visualizer import ViserVisualizer

def main(
    robot: str = "panda",                  # Robot to plan for
    planner: str = "rrtc",                 # Planner name to use (e.g., 'rrtc', 'prm', 'rrt')
    dataset: str = "problems.pkl",         # Pickled dataset to use
    problem: Union[str, List[str]] = [],   # Problem name or list of problems to evaluate
    trials: int = 1,                       # Number of trials to evaluate each instance
    print_failures: bool = False,          # Print out failures and invalid problems
    visualize: bool = False,              # Visualize solutions using Viser (any key=next, q=disable, w=skip problem set)
    pointcloud: bool = False,              # Use pointcloud rather than primitive geometry
    samples_per_object: int = 10000,       # If pointcloud, samples per object to use
    filter_radius: float = 0.02,           # Filter radius for pointcloud filtering
    filter_cull: bool = True,              # Cull pointcloud around robot by maximum distance
    planning_time: float = 1.0,            # Planning time limit in seconds
    sampler: str = "halton",
    **kwargs,
    ):

    if robot not in vamp.robots:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    (vamp_module, planner_func, plan_settings,
     simp_settings) = vamp.configure_robot_and_planner_with_kwargs(robot, planner, **kwargs)
    print(plan_settings)
    sampler = getattr(vamp_module, sampler)()
    vamp_folder = Path(__file__).parent.parent.parent / 'external' / 'vamp'
    problems_dir = vamp_folder / 'resources' / robot
    pickle_path = problems_dir / dataset
    
    
    # Check if pickle file exists, generate if not
    if not pickle_path.exists():
        print(f"Pickle file not found at {pickle_path}, generating from tar.bz2...")
        script_path = vamp_folder / 'resources' / 'problem_tar_to_pkl_json.py'
        result = subprocess.run([sys.executable, str(script_path), f'--robot={robot}'], check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate pickle file using {script_path}")
        print(f"Successfully generated pickle file")
    
    with open(pickle_path, 'rb') as f:
        problems = pickle.load(f)

    problem_names = list(problems['problems'].keys())
    if isinstance(problem, str):
        problem = [problem]

    if not problem:
        problem = problem_names
    else:
        for problem_name in problem:
            if problem_name not in problem_names:
                raise RuntimeError(
                    f"Problem `{problem_name}` not available! Available problems: {problem_names}"
                    )

    total_problems = 0
    valid_problems = 0
    failed_problems = 0
    
    # Initialize visualizer if requested
    vis = None
    visualize_enabled = visualize
    skip_to_next_problem_set = False
    
    if robot == "fetch":
        vis_camera_pos = [-0.5, -1.2, 1.5]
        vis_camera_target = [0.3, 0, 0.8]
    elif robot == "ur5":
        vis_camera_pos = [-1.1, -0.4, 2.0]
        vis_camera_target = [0.0, 0.4, 0.8]
    if visualize:
        vis = ViserVisualizer(robot_name=robot)
        # vis.add_grid()
        vis.set_camera(position=vis_camera_pos, target=vis_camera_target)
    tick = time.perf_counter()
    results = []
        
    for name, pset in problems['problems'].items():
        if name not in problem:
            continue
        
        # Reset skip flag for new problem set
        skip_to_next_problem_set = False

        failures = []
        invalids = []
        print(f"Evaluating {robot} on {name}: ")
        for i, data in tqdm(enumerate(pset)):
            total_problems += 1
            
            if i != 1:
                continue
            
            if not data['valid']:
                invalids.append(i)
                continue

            valid_problems += 1

            if pointcloud:
                r_min, r_max = vamp_module.min_max_radii()
                (env, original_pc, filtered_pc, filter_time, build_time) = vpc.problem_dict_to_pointcloud(
                    robot,
                    r_min,
                    r_max,
                    data,
                    samples_per_object,
                    filter_radius,
                    filter_cull,
                    )

                pointcloud_results = {
                    'original_pointcloud_size': len(original_pc),
                    'filtered_pointcloud_size': len(filtered_pc),
                    'filter_time': pd.Timedelta(nanoseconds = filter_time),
                    'capt_build_time': pd.Timedelta(nanoseconds = build_time)
                    }
            else:
                # env = vamp.problem_dict_to_vamp(data)
                env = vamp.Environment()
                # pos = (0.5, 0.3, 0.5)
                # orientation = (0, 0, 0)
                # half_extents = (0.2, 0.2, 0.1)
                # cuboid = vamp._core.Cuboid(pos, orientation, half_extents)
                # cuboid.name = "name"
                # env.add_cuboid(cuboid)

            sampler.reset()
            sampler.skip(0)
            for trial in range(trials):
                # Setup OMPL problem
                
                print(f"start: {data['start']}")
                result = planner_func(data['start'], data['goals'], env, plan_settings, sampler)
                if not result.solved:
                    failures.append(i)
                    break

                simple = vamp_module.simplify(result.path, env, simp_settings, sampler)
                
                simplified_path = simple.path
                
                # to array
                simplified_path_list = []
                n = len(simplified_path)
                for i in range(n):
                    simplified_path_list.append(list(simplified_path[i]))
                simplified_path = np.array(simplified_path_list)
                
                # interpolate with distance-based adaptive interpolation for smooth animation
                interp_path = []
                
                # Calculate distances between consecutive states
                distances = []
                for j in range(len(simplified_path) - 1):
                    dist = np.linalg.norm(simplified_path[j + 1] - simplified_path[j])
                    distances.append(dist)
                
                # Normalize distances and map to interpolation points
                max_distance = max(distances) if distances else 1.0
                min_points = 2  # minimum interpolation points per segment
                max_points = 20  # maximum interpolation points per segment
                
                for j in range(len(simplified_path) - 1):
                    # Scale number of points based on relative distance
                    relative_distance = distances[j] / max_distance if max_distance > 0 else 1.0
                    num_points = max(min_points, int(min_points + relative_distance * (max_points - min_points)))
                    
                    segment = np.linspace(simplified_path[j], simplified_path[j + 1], num=num_points, endpoint=False)
                    interp_path.extend(segment)
                
                # Don't forget the final state
                interp_path.append(simplified_path[-1])
                simplified_path = np.array(interp_path)

                # Visualization
                if visualize_enabled and vis is not None:
                    # Check if we should skip to next problem set
                    if skip_to_next_problem_set:
                        continue
                    # Clear previous visualization
                    vis.reset()
                    # vis.add_grid()
                    vis.set_camera(position=vis_camera_pos, target=vis_camera_target)
                    
                    # Load environment obstacles
                    vis.load_mbm_environment(data, padding=0.0, color=(0.8, 0.4, 0.2))
                    
                    if robot == "ur5":
                        # add box below 
                        vis.add_cylinder(position=[0, 0, 0.7], 
                                    radius = 0.25,
                                    length = 0.4,
                                    color=(0.4, 0.4, 0.4),
                                    name="base_cyl",
                                    )
                    # Convert path to numpy array
                    states = simplified_path
                    dimension = simplified_path.shape[1]
                    trajectory = np.array([list(state[0:dimension]) for state in states])
                    # interpolate
                    
                    
                    vis.visualize_trajectory(trajectory)
                    
                    print(f"\nVisualizing problem {name} [{i}] - Press any key to continue, 'q' to disable viz, 'w' to skip to next problem set")
                    key = vis.play_until_key_pressed(key='any', dt=0.05)
                    
                    if key == 'q':
                        print("Visualization disabled for remaining problems")
                        visualize_enabled = False
                    elif key == 'w':
                        print(f"Skipping to next problem set...")
                        skip_to_next_problem_set = True
                        break 
                    
                    # exit(0)
        
        

        failed_problems += len(failures)

        if print_failures:
            if invalids:
                print(f"  Invalid problems: {invalids}")

            if failures:
                print(f"  Failed on {failures}")

    tock = time.perf_counter()

    df = pd.DataFrame.from_dict(results)

    # Convert to microseconds
    df["planning_time"] = df["planning_time"].dt.microseconds
    df["simplification_time"] = df["simplification_time"].dt.microseconds
    df["avg_time_per_iteration"] = df["planning_iterations"] / df["planning_time"]

    # Pointcloud data
    if pointcloud:
        df["total_build_and_plan_time"] = df["total_time"] + df["filter_time"] + df["capt_build_time"]
        df["filter_time"] = df["filter_time"].dt.microseconds / 1e3
        df["capt_build_time"] = df["capt_build_time"].dt.microseconds / 1e3
        df["total_build_and_plan_time"] = df["total_build_and_plan_time"].dt.microseconds / 1e3

    df["total_time"] = df["total_time"].dt.microseconds

    # Get summary statistics
    time_stats = df[[
        "planning_time",
        "simplification_time",
        "total_time",
        "planning_iterations",
        "avg_time_per_iteration",
        ]].describe(percentiles = [0.25, 0.5, 0.75, 0.95])
    time_stats.drop(index = ["count"], inplace = True)

    cost_stats = df[[
        "initial_path_cost",
        "simplified_path_cost",
        ]].describe(percentiles = [0.25, 0.5, 0.75, 0.95])
    cost_stats.drop(index = ["count"], inplace = True)

    if pointcloud:
        pointcloud_stats = df[[
            "filter_time",
            "capt_build_time",
            "total_build_and_plan_time",
            ]].describe(percentiles = [0.25, 0.5, 0.75, 0.95])
        pointcloud_stats.drop(index = ["count"], inplace = True)

    print()
    print(
        tabulate(
            time_stats,
            headers = [
                'Planning Time (μs)',
                'Simplification Time (μs)',
                'Total Time (μs)',
                'Planning Iters.',
                'Time per Iter. (μs)',
                ],
            tablefmt = 'github'
            )
        )

    print(
        tabulate(
            cost_stats, headers = [
                ' Initial Cost (L2)',
                '    Simplified Cost (L2)',
                ], tablefmt = 'github'
            )
        )

    if pointcloud:
        print(
            tabulate(
                pointcloud_stats,
                headers = [
                    '  Filter Time (ms)',
                    '    CAPT Build Time (ms)',
                    'Total Time (ms)',
                    ],
                tablefmt = 'github'
                )
            )

    print(
        f"Solved / Valid / Total # Problems: {valid_problems - failed_problems} / {valid_problems} / {total_problems}"
        )
    print(f"Completed all problems in {df['total_time'].sum() / 1000:.3f} milliseconds")
    print(f"Total time including Python overhead: {(tock - tick) * 1000:.3f} milliseconds")


if __name__ == "__main__":
    Fire(main)
