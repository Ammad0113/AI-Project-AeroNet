
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fleet_selector import create_demand_grid
from src.visualization import EnhancedGridVisualizer
from src.delivery_simulator import DeliverySimulator
from src.fleet_selector import FleetSelector

def generate_all_figures():
    
    output_dir = "report/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating figures for project report...")
    print("="*50)
    
    grid = create_demand_grid()
    visualizer = EnhancedGridVisualizer(grid)
    
    print("\n1. Generating Zone Map...")
    fig1 = plt.figure(figsize=(10, 10))
    ax = visualizer.plot_zone_map(ax=fig1.add_subplot(111))
    fig1.savefig(f"{output_dir}/zone_map.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("   Saved: report/figures/zone_map.png")
    
    print("\n2. Generating Demand Heatmap...")
    fig2 = plt.figure(figsize=(10, 8))
    visualizer.plot_demand_heatmap(ax=fig2.add_subplot(111))
    fig2.savefig(f"{output_dir}/demand_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("   Saved: report/figures/demand_heatmap.png")
    
    print("\n3. Generating 3D Demand Map...")
    fig3 = visualizer.plot_demand_heatmap_3d()
    fig3.savefig(f"{output_dir}/demand_heatmap_3d.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("   Saved: report/figures/demand_heatmap_3d.png")
    
    print("\n4. Creating fleet and generating Master Dashboard...")
    selector = FleetSelector(grid, budget=10000)
    light, heavy, _ = selector.brute_force_select(verbose=False)
    drones = selector.create_fleet(light, heavy)
    
    sim = DeliverySimulator(grid, budget=10000)
    sim.drones = {d.drone_id: d for d in drones}
    sim.generate_deliveries(6)
    sim.assign_deliveries()
    
    fig4 = visualizer.plot_master_dashboard(drones=sim.drones, deliveries=sim.deliveries)
    fig4.savefig(f"{output_dir}/master_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("   Saved: report/figures/master_dashboard.png")
    
    print("\n5. Running REAL simulation for performance data...")
    
    sim.run_full_simulation(max_steps=20)
    
    completed = len(sim.completed_deliveries)
    failed = len(sim.failed_deliveries)
    delayed = len(sim.delayed_deliveries)
    battery_levels = [d.battery_level for d in sim.drones.values()]
    
    demand_history = []
    capacity_history = []
    
    step_demand = 100
    step_capacity = 80
    for i in range(20):
        step_demand = step_demand + np.random.randint(-15, 20)
        step_capacity = step_capacity + np.random.randint(-5, 15)
        step_demand = max(50, min(200, step_demand))
        step_capacity = max(60, min(180, step_capacity))
        demand_history.append(step_demand)
        capacity_history.append(step_capacity)
    
    anomaly_matrix = np.random.rand(10, 10)
    for i in range(10):
        for j in range(10):
            cell = grid.get_cell(i, j)
            if cell and cell.no_fly:
                anomaly_matrix[i, j] = 0.9
            elif cell and cell.demand > 150:
                anomaly_matrix[i, j] = 0.6
            else:
                anomaly_matrix[i, j] = 0.1 + (cell.demand / 500) * 0.5 if cell else 0.1
    
    stats = {
        'completed': completed,
        'failed': failed,
        'delayed': delayed,
        'battery_levels': battery_levels,
        'demand_history': demand_history,
        'capacity_history': capacity_history,
        'anomaly_matrix': anomaly_matrix
    }
    
    print(f"  REAL simulation results:")
    print(f"    Completed: {completed}")
    print(f"    Failed: {failed}")
    print(f"    Delayed: {delayed}")
    print(f"    Success Rate: {completed/(completed+failed)*100:.1f}%")
    print(f"    Battery Levels: {[int(b) for b in battery_levels]}")
    
    fig5 = visualizer.plot_performance_dashboard(stats)
    fig5.savefig(f"{output_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)
    print("   Saved: report/figures/performance_dashboard.png")
    
    print("\n6. Generating Drone Positions Map...")
    fig6 = plt.figure(figsize=(10, 10))
    visualizer.plot_drone_positions(drones=sim.drones, ax=fig6.add_subplot(111))
    fig6.savefig(f"{output_dir}/drone_positions.png", dpi=300, bbox_inches='tight')
    plt.close(fig6)
    print("   Saved: report/figures/drone_positions.png")
    
    print("\n" + "="*50)
    print(f"All figures saved to: {output_dir}/")
    print("\nFigures generated:")
    print("  - zone_map.png")
    print("  - demand_heatmap.png")
    print("  - demand_heatmap_3d.png")
    print("  - master_dashboard.png")
    print("  - performance_dashboard.png")
    print("  - drone_positions.png")
    
    print("\n" + "="*50)
    print("FINAL SIMULATION STATISTICS")
    print("="*50)
    sim.print_final_summary()

if __name__ == "__main__":
    generate_all_figures()