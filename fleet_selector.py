"""
fleet_selector.py - Fleet Selection for AeroNet Lite
"""

import random
import pandas as pd
import os
import zipfile
from typing import List, Tuple
from dataclasses import dataclass
from grid_model import AeroNetGrid, DroneType, Drone, ZoneType


@dataclass
class DroneConfig:
    drone_type: DroneType
    cost: int
    payload_kg: int
    range_cells: int
    max_count: int = 10


class FleetSelector:
    
    DRONE_SPECS = {
        DroneType.LIGHT: DroneConfig(
            drone_type=DroneType.LIGHT,
            cost=1000,
            payload_kg=2,
            range_cells=12,
            max_count=10
        ),
        DroneType.HEAVY: DroneConfig(
            drone_type=DroneType.HEAVY,
            cost=1800,
            payload_kg=5,
            range_cells=20,
            max_count=6
        )
    }
    
    BUDGET = 10000
    
    def __init__(self, grid: AeroNetGrid, budget: int = None):
        self.grid = grid
        self.budget = budget or self.BUDGET
        
    def calculate_total_demand(self) -> float:
        total_demand = 0
        for row in range(self.grid.size):
            for col in range(self.grid.size):
                cell = self.grid.get_cell(row, col)
                if cell:
                    if cell.zone == ZoneType.RESIDENTIAL:
                        total_demand += cell.demand * 0.1
                    elif cell.zone == ZoneType.COMMERCIAL:
                        total_demand += cell.demand * 0.15
                    else:
                        total_demand += cell.demand * 0.05
        return max(total_demand, 100)
    
    def calculate_delivery_capacity(self, light_count: int, heavy_count: int) -> float:
        light_capacity = light_count * self.DRONE_SPECS[DroneType.LIGHT].payload_kg * 4
        heavy_capacity = heavy_count * self.DRONE_SPECS[DroneType.HEAVY].payload_kg * 4
        return light_capacity + heavy_capacity
    
    def calculate_demand_coverage(self, light_count: int, heavy_count: int) -> float:
        total_demand = self.calculate_total_demand()
        delivery_capacity = self.calculate_delivery_capacity(light_count, heavy_count)
        effective_capacity = delivery_capacity * 50
        
        if total_demand == 0:
            return 1.0
        
        coverage = min(1.0, effective_capacity / total_demand)
        return coverage
    
    def calculate_fitness(self, light_count: int, heavy_count: int) -> float:
        light_cost = light_count * self.DRONE_SPECS[DroneType.LIGHT].cost
        heavy_cost = heavy_count * self.DRONE_SPECS[DroneType.HEAVY].cost
        total_cost = light_cost + heavy_cost
        
        if total_cost > self.budget:
            return -1.0
        
        coverage = self.calculate_demand_coverage(light_count, heavy_count)
        budget_used_ratio = total_cost / self.budget
        fitness = coverage * 1.0 + budget_used_ratio * 0.2
        
        return fitness
    
    def brute_force_select(self, verbose: bool = True) -> Tuple[int, int, float]:
        if verbose:
            print("\n" + "="*60)
            print("FLEET SELECTION - BRUTE FORCE METHOD")
            print("="*60)
            print(f"Budget: ${self.budget}")
            print(f"Total Demand: {self.calculate_total_demand():.0f} units")
        
        best_light, best_heavy = 0, 0
        best_fitness = -float('inf')
        
        max_light = min(10, self.budget // 1000)
        max_heavy = min(6, self.budget // 1800)
        
        valid_configs = []
        
        for light_count in range(max_light + 1):
            for heavy_count in range(max_heavy + 1):
                fitness = self.calculate_fitness(light_count, heavy_count)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_light, best_heavy = light_count, heavy_count
                
                if fitness >= 0:
                    cost = light_count * 1000 + heavy_count * 1800
                    coverage = self.calculate_demand_coverage(light_count, heavy_count)
                    valid_configs.append((light_count, heavy_count, cost, coverage, fitness))
        
        if verbose and valid_configs:
            valid_configs.sort(key=lambda x: x[4], reverse=True)
            print(f"\nTop 10 Fleet Configurations:")
            print("-" * 65)
            print(f"  {'Light':<6} {'Heavy':<6} {'Cost':<8} {'Coverage':<10} {'Fitness':<8}")
            print("-" * 65)
            for l, h, cost, cov, fit in valid_configs[:10]:
                print(f"  {l:<6} {h:<6} ${cost:<7} {cov:<9.1%} {fit:<8.4f}")
        
        self._print_result(best_light, best_heavy, best_fitness)
        return best_light, best_heavy, best_fitness
    
    def genetic_algorithm_select(
        self,
        population_size: int = 60,
        generations: int = 80,
        mutation_rate: float = 0.2,
        verbose: bool = True
    ) -> Tuple[int, int, float]:
        if verbose:
            print("\n" + "="*60)
            print("FLEET SELECTION - GENETIC ALGORITHM")
            print("="*60)
            print(f"Budget: ${self.budget}")
            print(f"Total Demand: {self.calculate_total_demand():.0f} units")
        
        max_light = min(10, self.budget // 1000)
        max_heavy = min(6, self.budget // 1800)
        
        population = []
        
        for light in [2, 3, 4, 5, 6]:
            for heavy in [1, 2, 3, 4]:
                if light * 1000 + heavy * 1800 <= self.budget:
                    population.append([light, heavy])
        
        while len(population) < population_size:
            light = random.randint(0, max_light)
            max_h_for_budget = (self.budget - light * 1000) // 1800
            heavy = random.randint(0, min(max_heavy, max(0, max_h_for_budget)))
            population.append([light, heavy])
        
        best_light, best_heavy = 0, 0
        best_fitness = -float('inf')
        
        for generation in range(generations):
            fitnesses = [self.calculate_fitness(l, h) for l, h in population]
            
            for i, fit in enumerate(fitnesses):
                if fit > best_fitness:
                    best_fitness = fit
                    best_light, best_heavy = population[i]
            
            if verbose and generation % 20 == 0:
                valid_fits = [f for f in fitnesses if f >= 0]
                if valid_fits:
                    print(f"  Gen {generation:3d}: Best fitness = {best_fitness:.4f} ({best_light}L+{best_heavy}H)")
            
            new_population = []
            
            sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            for i in range(min(5, len(sorted_idx))):
                if fitnesses[sorted_idx[i]] >= 0:
                    new_population.append(population[sorted_idx[i]].copy())
            
            while len(new_population) < population_size:
                p1_idx = random.randint(0, len(population)-1)
                p2_idx = random.randint(0, len(population)-1)
                for _ in range(2):
                    idx = random.randint(0, len(population)-1)
                    if fitnesses[idx] > fitnesses[p1_idx]:
                        p1_idx = idx
                    if fitnesses[idx] > fitnesses[p2_idx]:
                        p2_idx = idx
                
                child = [
                    random.choice([population[p1_idx][0], population[p2_idx][0]]),
                    random.choice([population[p1_idx][1], population[p2_idx][1]])
                ]
                
                if random.random() < mutation_rate:
                    child[0] = random.randint(0, max_light)
                    max_h = (self.budget - child[0] * 1000) // 1800
                    child[1] = random.randint(0, min(max_heavy, max(0, max_h)))
                
                new_population.append(child)
            
            population = new_population
        
        if verbose:
            print(f"\nBest solution: {best_light} Light + {best_heavy} Heavy drones")
            self._print_result(best_light, best_heavy, best_fitness)
        
        return best_light, best_heavy, best_fitness
    
    def _print_result(self, light_count: int, heavy_count: int, fitness: float):
        light_cost = light_count * 1000
        heavy_cost = heavy_count * 1800
        total_cost = light_cost + heavy_cost
        
        coverage = self.calculate_demand_coverage(light_count, heavy_count)
        total_demand = self.calculate_total_demand()
        delivery_capacity = self.calculate_delivery_capacity(light_count, heavy_count)
        
        print("\n" + "-"*50)
        print("FLEET SELECTION RESULT")
        print("-"*50)
        print(f"  Light Drones:  {light_count} @ $1000 = ${light_cost}")
        print(f"  Heavy Drones:  {heavy_count} @ $1800 = ${heavy_cost}")
        print(f"  Total Cost:    ${total_cost}")
        print(f"  Budget Left:   ${self.budget - total_cost}")
        print(f"  Total Demand:  {total_demand:.0f} units")
        print(f"  Delivery Capacity: {delivery_capacity * 50:.0f} units/day")
        print(f"  Coverage:      {coverage:.1%}")
        print(f"  Fitness Score: {fitness:.4f}")
        
        if coverage >= 0.7:
            print("\n  Excellent fleet configuration!")
        elif coverage >= 0.5:
            print("\n  Good fleet configuration")
        elif coverage >= 0.3:
            print("\n  Moderate fleet - consider adding more drones")
        else:
            print("\n  Low coverage - increase budget or add more hubs")
    
    def create_fleet(self, light_count: int, heavy_count: int) -> List[Drone]:
        drones = []
        drone_id = 1
        
        hubs = self.grid.get_hub_locations()
        hub_location = hubs[0] if hubs else (0, 0)
        
        for i in range(light_count):
            drones.append(Drone(
                drone_id=f"D{drone_id:03d}",
                drone_type=DroneType.LIGHT,
                current_location=hub_location,
                battery_level=100.0
            ))
            drone_id += 1
        
        for i in range(heavy_count):
            drones.append(Drone(
                drone_id=f"D{drone_id:03d}",
                drone_type=DroneType.HEAVY,
                current_location=hub_location,
                battery_level=100.0
            ))
            drone_id += 1
        
        return drones


def load_population_density_data():
    """Load real population density dataset by extracting zip (same as load_datasets.py)"""
    
    zip_path = "data/archive (4).zip"
    extract_dir = "extracted_data/population_density"
    
    if os.path.exists(zip_path):
        print(f"Found zip file: {zip_path}")
        
        if not os.path.exists(extract_dir):
            print(f"Extracting {zip_path} to {extract_dir}...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            print("Extraction complete")
        
        csv_path = os.path.join(extract_dir, "uscitypopdensity.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded population density data from {csv_path}")
            print(f"Shape: {df.shape}")
            return df
        else:
            print(f"CSV not found at {csv_path}")
            for file in os.listdir(extract_dir):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(extract_dir, file))
                    print(f"Found alternative CSV: {file}")
                    return df
    else:
        print(f"Zip not found: {zip_path}")
        print(f"Current working directory: {os.getcwd()}")
    
    print("Using synthetic density values (fallback)")
    return None


def create_demand_grid() -> AeroNetGrid:
    from grid_model import create_sample_grid
    
    grid = create_sample_grid()
    
    hub_locations = [(0, 0), (0, 4), (2, 2), (5, 5), (8, 8)]
    for row, col in hub_locations:
        grid.set_cell(row, col, is_hub=True, is_charging=True)
    
    pop_df = load_population_density_data()
    
    if pop_df is not None:
        print("Applying real population density data to grid...")
        
        if 'pop_density' in pop_df.columns:
            density_values = pop_df['pop_density'].dropna().values
        else:
            numeric_cols = pop_df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                density_values = pop_df[numeric_cols[0]].dropna().values
            else:
                density_values = None
        
        if density_values is not None and len(density_values) > 0:
            min_density = density_values.min()
            max_density = density_values.max()
            print(f"Density range: {min_density:.0f} to {max_density:.0f}")
            
            for row in range(10):
                for col in range(10):
                    cell = grid.get_cell(row, col)
                    if cell:
                        idx = (row * 10 + col) % len(density_values)
                        real_density = density_values[idx]
                        
                        if cell.zone == ZoneType.RESIDENTIAL:
                            cell.demand = 50 + (real_density / max_density) * 200
                            cell.density = int(real_density)
                        elif cell.zone == ZoneType.COMMERCIAL:
                            cell.demand = 80 + (real_density / max_density) * 250
                            cell.density = int(real_density)
                        elif cell.zone == ZoneType.HOSPITAL:
                            cell.demand = 30 + (real_density / max_density) * 100
                            cell.density = int(real_density)
                        elif cell.zone == ZoneType.SCHOOL:
                            cell.demand = 25 + (real_density / max_density) * 80
                            cell.density = int(real_density)
                        else:
                            cell.demand = 10 + (real_density / max_density) * 50
                            cell.density = int(real_density)
                        
                        cell.demand = max(10, min(500, cell.demand))
            
            print("Real population density applied to grid")
            return grid
    
    print("Using synthetic demand values (fallback)")
    for row in range(10):
        for col in range(10):
            cell = grid.get_cell(row, col)
            if cell:
                if cell.zone == ZoneType.RESIDENTIAL:
                    cell.demand = 150
                elif cell.zone == ZoneType.COMMERCIAL:
                    cell.demand = 250
                elif cell.zone == ZoneType.HOSPITAL:
                    cell.demand = 80
                elif cell.zone == ZoneType.SCHOOL:
                    cell.demand = 60
                else:
                    cell.demand = 30
                cell.density = 5000
    
    return grid


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AERONET LITE - FLEET SELECTOR MODULE")
    print("="*60)
    
    grid = create_demand_grid()
    selector = FleetSelector(grid, budget=10000)
    
    print("\nGRID STATISTICS:")
    print(f"  Total Hubs: {len(grid.get_hub_locations())}")
    print(f"  Total Demand: {selector.calculate_total_demand():.0f} units")
    
    print("\nSample cell demands:")
    for r in range(3):
        for c in range(3):
            cell = grid.get_cell(r, c)
            if cell:
                print(f"  Cell ({r},{c}): Zone={cell.zone.value}, Demand={cell.demand:.0f}, Density={cell.density}")
    
    light_bf, heavy_bf, fitness_bf = selector.brute_force_select(verbose=True)
    light_ga, heavy_ga, fitness_ga = selector.genetic_algorithm_select(verbose=True)
    
    print("\n" + "="*60)
    print("CREATING DRONE FLEET")
    print("="*60)
    
    drones = selector.create_fleet(light_bf, heavy_bf)
    
    if drones:
        print(f"\n  Created fleet with {len(drones)} drones:")
        for drone in drones:
            print(f"     - {drone.drone_id}: {drone.drone_type.value} Drone")
        print(f"\n  Total fleet cost: ${light_bf*1000 + heavy_bf*1800}")
        print(f"  Demand coverage: {selector.calculate_demand_coverage(light_bf, heavy_bf):.1%}")
    else:
        print("\n  No drones selected. Debug info:")
        print(f"     Budget: ${selector.budget}")
        print(f"     Light drone cost: $1000, Heavy: $1800")