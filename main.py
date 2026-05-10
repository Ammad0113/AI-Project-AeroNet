import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from grid_model import create_sample_grid
from layout_validator import LayoutValidator
from fleet_selector import FleetSelector, create_demand_grid
from astar_planner import AStarPlanner
from delivery_simulator import DeliverySimulator
from ml_pipeline import MLPipeline
from visualization import EnhancedGridVisualizer


def print_banner():
    print("\n" + "="*70)
    print("     AERONET LITE - AUTONOMOUS DRONE DELIVERY SYSTEM")
    print("     BS Data Science AI Semester Project SP2026")
    print("="*70)


def run_full_simulation():
    print("\n[1] Running Full Simulation")
    print("-"*50)
    
    grid = create_demand_grid()
    
    if not grid.get_medical_pickup_locations():
        grid.set_cell(2, 2, is_medical_pickup=True)
    
    simulator = DeliverySimulator(grid, budget=10000)
    simulator.run_full_simulation(max_steps=20)
    
    return simulator


def run_layout_validation():
    print("\n[2] Running Layout Validation (CSP)")
    print("-"*50)
    
    grid = create_demand_grid()
    validator = LayoutValidator(grid)
    result = validator.validate_all(verbose=True)
    
    return result


def run_fleet_selection():
    print("\n[3] Running Fleet Selection")
    print("-"*50)
    
    grid = create_demand_grid()
    selector = FleetSelector(grid, budget=10000)
    
    print("\n--- Brute Force Method ---")
    light_bf, heavy_bf, fitness_bf = selector.brute_force_select(verbose=True)
    
    print("\n--- Genetic Algorithm Method ---")
    light_ga, heavy_ga, fitness_ga = selector.genetic_algorithm_select(verbose=True)
    
    return (light_bf, heavy_bf, fitness_bf), (light_ga, heavy_ga, fitness_ga)


def run_astar_demo():
    print("\n[4] Running A* Path Planner Demo")
    print("-"*50)
    
    grid = create_demand_grid()
    planner = AStarPlanner(grid)
    
    start = (0, 0)
    goal = (9, 9)
    
    path, cost, success = planner.find_path(start, goal, verbose=True)
    
    if success:
        print(f"Path found! Length: {len(path)} cells, Cost: {cost:.2f}")
    
    return path, cost, success


def run_ml_pipeline():
    print("\n[5] Running ML Pipeline")
    print("-"*50)
    
    pipeline = MLPipeline()
    pipeline.load_models(verbose=True)
    
    return pipeline


def show_visualization():
    print("\n[6] Generating Visualizations")
    print("-"*50)
    
    grid = create_demand_grid()
    visualizer = EnhancedGridVisualizer(grid)
    
    print("Creating Master Dashboard...")
    fig1 = visualizer.plot_master_dashboard()
    
    print("Creating 3D Demand Map...")
    fig2 = visualizer.plot_demand_heatmap_3d()
    
    plt.show()
    
    print("Visualizations complete")
    
    return fig1, fig2


def main():
    print_banner()
    
    print("\n" + "="*70)
    print("SELECT SIMULATION MODE")
    print("="*70)
    print("1. Run Full 20-Step Simulation")
    print("2. Run Layout Validation Only (CSP)")
    print("3. Run Fleet Selection Only (GA + Brute Force)")
    print("4. Run A* Path Planner Demo")
    print("5. Run ML Pipeline Demo")
    print("6. Show Visualizations Only")
    print("7. Run ALL (Complete System Test)")
    print("8. Exit")
    
    choice = input("\nEnter your choice (1-8): ").strip()
    
    if choice == "1":
        run_full_simulation()
    
    elif choice == "2":
        run_layout_validation()
    
    elif choice == "3":
        run_fleet_selection()
    
    elif choice == "4":
        run_astar_demo()
    
    elif choice == "5":
        run_ml_pipeline()
    
    elif choice == "6":
        show_visualization()
    
    elif choice == "7":
        print("\n" + "="*70)
        print("RUNNING COMPLETE SYSTEM TEST")
        print("="*70)
        
        run_layout_validation()
        run_fleet_selection()
        run_astar_demo()
        run_ml_pipeline()
        run_full_simulation()
        
        print("\n" + "="*70)
        print("SYSTEM TEST COMPLETE - ALL MODULES WORKING")
        print("="*70)
    
    elif choice == "8":
        print("Exiting")
        return
    
    else:
        print("Invalid choice! Running full simulation by default")
        run_full_simulation()


if __name__ == "__main__":
    main()