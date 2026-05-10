"""
delivery_simulator.py - Main Delivery Simulator for AeroNet Lite
"""

import random
import time
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

from grid_model import (
    AeroNetGrid, Drone, DroneType, Delivery, ZoneType,
    create_sample_grid
)
from layout_validator import LayoutValidator
from fleet_selector import FleetSelector, create_demand_grid
from astar_planner import AStarPlanner
from ml_pipeline import MLPipeline


class DeliverySimulator:
    
    def __init__(self, grid: AeroNetGrid = None, budget: int = 10000):
        self.grid = grid or create_demand_grid()
        self.budget = budget
        self.planner = AStarPlanner(self.grid)
        self.ml_pipeline = MLPipeline()
        self.drones: Dict[str, Drone] = {}
        self.deliveries: List[Delivery] = []
        self.completed_deliveries: List[Delivery] = []
        self.failed_deliveries: List[Delivery] = []
        self.delayed_deliveries: List[Delivery] = []
        
        self.step = 0
        self.event_log: List[str] = []
        self.next_delivery_id = 1
        
        self.drone_path_history: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.drone_battery_history: Dict[str, List[float]] = defaultdict(list)
        
    def log_event(self, message: str, level: str = "INFO"):
        log_entry = f"[Step {self.step}] [{level}] {message}"
        self.event_log.append(log_entry)
        print(log_entry)
    
    def initialize_simulation(self, verbose: bool = True):
        self.log_event("="*50)
        self.log_event("SIMULATION INITIALIZATION")
        self.log_event("="*50)
        
        self.log_event("Step 1: Running layout validation (CSP)...")
        validator = LayoutValidator(self.grid)
        layout_valid = validator.validate_all(verbose=verbose)
        
        if not layout_valid:
            self.log_event("WARNING: Layout validation failed! Some rules violated.", "WARNING")
            self.log_event("Suggestions to fix:")
            for suggestion in validator.get_suggestions()[:3]:
                self.log_event(f"  - {suggestion}")
        else:
            self.log_event("Layout validation passed!")
        
        self.log_event("\nStep 2-3: Selecting drone fleet...")
        selector = FleetSelector(self.grid, self.budget)
        
        light_count, heavy_count, fitness = selector.brute_force_select(verbose=False)
        
        self.drones = {}
        drone_list = selector.create_fleet(light_count, heavy_count)
        for drone in drone_list:
            self.drones[drone.drone_id] = drone
        
        total_drones = len(self.drones)
        self.log_event(f"Fleet selected: {light_count} Light + {heavy_count} Heavy = {total_drones} total drones")
        self.log_event(f"Total cost: ${light_count*1000 + heavy_count*1800}")
        self.log_event(f"Fitness score: {fitness:.4f}")
        
        self.log_event("\nLoading ML models for demand forecasting and anomaly detection...")
        self.ml_pipeline.load_models(verbose=verbose)
        
        return layout_valid
    
    def generate_deliveries(self, count: int = 5):
        self.log_event(f"\nGenerating {count} delivery requests...")
        
        hubs = self.grid.get_hub_locations()
        if not hubs:
            self.log_event("ERROR: No hubs available for deliveries!", "ERROR")
            return
        
        medical_pickups = self.grid.get_medical_pickup_locations()
        residential_cells = self.grid.get_cells_by_zone(ZoneType.RESIDENTIAL)
        commercial_cells = self.grid.get_cells_by_zone(ZoneType.COMMERCIAL)
        
        current_hour = self.step % 24
        
        for i in range(count):
            delivery_type = random.choice(["medical", "residential"])
            
            if delivery_type == "medical" and medical_pickups:
                pickup = random.choice(medical_pickups)
                dropoff = random.choice(residential_cells or commercial_cells or [(5, 5)])
                weight = random.uniform(0.5, 1.5)
                priority = 1
            else:
                pickup = random.choice(hubs)
                dropoff = random.choice(residential_cells or commercial_cells or [(5, 5)])
                weight = random.uniform(0.5, 1.0)
                priority = 2
            
            delivery = Delivery(
                delivery_id=self.next_delivery_id,
                pickup_location=pickup,
                dropoff_location=dropoff,
                weight_kg=weight,
                priority=priority,
                status="pending"
            )
            self.deliveries.append(delivery)
            self.next_delivery_id += 1
            
            predicted_demand = self.ml_pipeline.predict_demand(hour=current_hour)
            
            self.log_event(f"  Delivery {delivery.delivery_id}: {delivery_type.upper()} "
                          f"({pickup} -> {dropoff}) {weight:.1f}kg, Priority {priority}, "
                          f"Predicted demand: {predicted_demand:.0f}")
    
    def assign_deliveries(self):
        self.log_event("\nAssigning deliveries to drones...")
        
        pending = [d for d in self.deliveries if d.status == "pending"]
        available = [d for d in self.drones.values() if d.status == "idle"]
        
        pending.sort(key=lambda x: x.priority)
        
        assigned = 0
        for delivery in pending[:len(available)]:
            drone = available[assigned]
            
            if not drone.can_carry(delivery.weight_kg):
                self.log_event(f"  Drone {drone.drone_id} cannot carry {delivery.weight_kg}kg", "WARNING")
                continue
            
            hub = drone.current_location
            route, cost, success = self.planner.plan_delivery_route(
                hub, delivery.pickup_location, delivery.dropoff_location,
                verbose=False
            )
            
            if success:
                drone.assigned_delivery = delivery
                drone.route = route
                drone.route_index = 0
                drone.status = "moving"
                delivery.assigned_drone_id = drone.drone_id
                delivery.status = "assigned"
                assigned += 1
                
                self.log_event(f"  Delivery {delivery.delivery_id} -> Drone {drone.drone_id} "
                              f"(Route length: {len(route)} cells)")
            else:
                self.log_event(f"  No route for Delivery {delivery.delivery_id}!", "ERROR")
                delivery.status = "failed"
                self.failed_deliveries.append(delivery)
        
        if assigned == 0:
            self.log_event("  No deliveries assigned")
    
    def update_drone_positions(self, moves_per_step: int = 3):
        for drone in self.drones.values():
            if drone.status != "moving":
                continue
            
            for _ in range(moves_per_step):
                self.drone_path_history[drone.drone_id].append(drone.current_location)
                self.drone_battery_history[drone.drone_id].append(drone.battery_level)
                
                if drone.route_index < len(drone.route) - 1:
                    drone.route_index += 1
                    drone.current_location = drone.route[drone.route_index]
                    drone.battery_level -= 2.0
                    
                    if drone.battery_level <= 0:
                        self.log_event(f"  Drone {drone.drone_id} out of battery!", "ERROR")
                        drone.status = "failed"
                        if drone.assigned_delivery:
                            drone.assigned_delivery.status = "failed"
                            self.failed_deliveries.append(drone.assigned_delivery)
                        break
                else:
                    if drone.assigned_delivery:
                        delivery = drone.assigned_delivery
                        delivery.status = "completed"
                        self.completed_deliveries.append(delivery)
                        if delivery in self.deliveries:
                            self.deliveries.remove(delivery)
                        
                        self.log_event(f"  Drone {drone.drone_id} completed Delivery {delivery.delivery_id}")
                        
                        drone.assigned_delivery = None
                        drone.route = []
                        drone.route_index = 0
                        drone.status = "idle"
                        if self.grid.get_hub_locations():
                            drone.current_location = self.grid.get_hub_locations()[0]
                    break
    
    def activate_no_fly(self, row: int, col: int):
        self.log_event(f"\nDISRUPTION: No-fly cell activated at ({row}, {col})!")
        
        cell = self.grid.get_cell(row, col)
        if cell:
            cell.no_fly = True
        
        affected = []
        for drone in self.drones.values():
            if drone.status == "moving" and drone.route:
                if (row, col) in drone.route[drone.route_index:]:
                    affected.append(drone)
        
        if affected:
            self.log_event(f"  Affected drones: {[d.drone_id for d in affected]}")
            
            for drone in affected:
                self.reroute_drone(drone, (row, col))
    
    def reroute_drone(self, drone: Drone, blocked_cell: Tuple[int, int]):
        self.log_event(f"  Rerouting Drone {drone.drone_id}...")
        
        delivery = drone.assigned_delivery
        if not delivery:
            return
        
        current_pos = drone.current_location
        
        if delivery.status == "assigned":
            target = delivery.pickup_location
        else:
            target = delivery.dropoff_location
        
        path, cost, success = self.planner.find_alternative_path(
            current_pos, target, blocked_cells={blocked_cell}, verbose=False
        )
        
        if success:
            remaining_route = path[1:]
            
            if target == delivery.pickup_location:
                route2, cost2, success2 = self.planner.find_path(
                    delivery.pickup_location, delivery.dropoff_location
                )
                route3, cost3, success3 = self.planner.find_path(
                    delivery.dropoff_location, self.grid.get_hub_locations()[0]
                )
                if success2 and success3:
                    drone.route = remaining_route + route2[1:] + route3[1:]
                else:
                    self.log_event(f"  No alternative route for Drone {drone.drone_id}", "ERROR")
                    drone.status = "failed"
                    delivery.status = "failed"
                    self.failed_deliveries.append(delivery)
                    return
            else:
                route3, cost3, success3 = self.planner.find_path(
                    delivery.dropoff_location, self.grid.get_hub_locations()[0]
                )
                if success3:
                    drone.route = remaining_route + route3[1:]
                else:
                    self.log_event(f"  No alternative route for Drone {drone.drone_id}", "ERROR")
                    drone.status = "failed"
                    delivery.status = "failed"
                    self.failed_deliveries.append(delivery)
                    return
            
            drone.route_index = 0
            self.log_event(f"  Drone {drone.drone_id} rerouted successfully")
        else:
            self.log_event(f"  No alternative route found! Delivery {delivery.delivery_id} failed.", "ERROR")
            drone.status = "failed"
            delivery.status = "failed"
            self.failed_deliveries.append(delivery)
    
    def detect_anomaly_ml(self, drone: Drone) -> Optional[Tuple[str, float, str]]:
        
        if self.ml_pipeline.anomaly_model is None:
            return None
        
        battery_history = self.drone_battery_history.get(drone.drone_id, [])
        battery_drop = 0
        if len(battery_history) >= 2:
            battery_drop = max(0, battery_history[-2] - battery_history[-1])
        
        route_deviation = 0
        if drone.route and drone.route_index < len(drone.route):
            expected = drone.route[drone.route_index]
            actual = drone.current_location
            route_deviation = abs(expected[0] - actual[0]) + abs(expected[1] - actual[1])
        
        altitude_change = 0
        if len(self.drone_path_history.get(drone.drone_id, [])) >= 2:
            prev_pos = self.drone_path_history[drone.drone_id][-2]
            curr_pos = drone.current_location
            altitude_change = curr_pos[0] - prev_pos[0]
        
        anomaly_type, confidence, recommendation = self.ml_pipeline.analyze_drone_telemetry(
            drone_id=drone.drone_id,
            battery_level=drone.battery_level,
            current_speed=10.0,
            altitude_change=altitude_change,
            route_deviation=route_deviation
        )
        
        if anomaly_type != "truth" and confidence > 0.6:
            return anomaly_type, confidence, recommendation
        
        return None
    
    def run_simulation_step(self):
        self.step += 1
        self.log_event(f"\n{'='*50}")
        self.log_event(f"STEP {self.step}")
        self.log_event(f"{'='*50}")
        
        if self.step == 2:
            self.generate_deliveries(4)
            self.assign_deliveries()
        
        elif 3 <= self.step <= 8:
            self.update_drone_positions(moves_per_step=3)
            if self.step == 8:
                active = sum(1 for d in self.drones.values() if d.status == "moving")
                self.log_event(f"  Active deliveries: {active}")
                self.log_event(f"  Completed so far: {len(self.completed_deliveries)}")
        
        elif self.step == 9:
            self.activate_no_fly(4, 7)
            self.update_drone_positions(moves_per_step=3)
        
        elif 10 <= self.step <= 13:
            self.update_drone_positions(moves_per_step=3)
        
        elif self.step == 14:
            self.generate_deliveries(2)
            self.assign_deliveries()
            self.update_drone_positions(moves_per_step=3)
        
        elif 15 <= self.step <= 17:
            self.update_drone_positions(moves_per_step=3)
        
        elif self.step == 18:
            self.update_drone_positions(moves_per_step=3)
            for drone in self.drones.values():
                if drone.status == "moving":
                    drone.battery_level -= 20
                    self.log_event(f"  Battery drain injected for {drone.drone_id}")
                    anomaly_result = self.detect_anomaly_ml(drone)
                    if anomaly_result:
                        anomaly_type, confidence, recommendation = anomaly_result
                        self.log_event(f"  ML ANOMALY DETECTED: {anomaly_type} (confidence: {confidence:.1%})", "WARNING")
                        self.log_event(f"  Recommendation: {recommendation}")
                    break
        
        elif self.step == 19:
            self.update_drone_positions(moves_per_step=3)
            for drone in self.drones.values():
                if drone.battery_level < 40:
                    self.log_event(f"  Drone {drone.drone_id} returning to hub")
                    hub = self.grid.get_hub_locations()[0]
                    path, _, success = self.planner.find_path(drone.current_location, hub, verbose=False)
                    if success:
                        drone.route = path
                        drone.route_index = 0
                        drone.is_returning_to_hub = True
                    break
        
        elif self.step == 20:
            self.update_drone_positions(moves_per_step=5)
            self.print_final_summary()
        
        else:
            self.update_drone_positions(moves_per_step=3)
    
    def print_final_summary(self):
        self.log_event("\n" + "="*60)
        self.log_event("SIMULATION COMPLETE - FINAL SUMMARY")
        self.log_event("="*60)
        
        completed = len(self.completed_deliveries)
        failed = len(self.failed_deliveries)
        pending = len([d for d in self.deliveries if d.status == "pending"])
        in_progress = len([d for d in self.deliveries if d.status == "assigned"])
        
        self.log_event(f"\nDELIVERY STATISTICS:")
        self.log_event(f"  Completed: {completed}")
        self.log_event(f"  In Progress: {in_progress}")
        self.log_event(f"  Failed: {failed}")
        self.log_event(f"  Pending: {pending}")
        self.log_event(f"  Total: {completed + failed + pending + in_progress}")
        
        self.log_event(f"\nDRONE STATUS:")
        for drone in self.drones.values():
            icon = "IDLE" if drone.status == "idle" else "MOVING" if drone.status == "moving" else "FAIL"
            self.log_event(f"  {icon} {drone.drone_id}: Battery {drone.battery_level:.0f}%")
        
        total = completed + failed
        if total > 0:
            success_rate = (completed / total) * 100
            self.log_event(f"\nSUCCESS RATE: {success_rate:.1f}%")
        
        self.log_event(f"\nDELIVERY TIMELINE:")
        for d in self.completed_deliveries[:5]:
            self.log_event(f"  Delivery {d.delivery_id}: COMPLETED")
        for d in self.failed_deliveries[:3]:
            self.log_event(f"  Delivery {d.delivery_id}: FAILED")
        
    def run_full_simulation(self, max_steps: int = 20):
        print("\n" + "="*70)
        print("AERONET LITE - DRONE DELIVERY SIMULATION")
        print("="*70)
        
        self.initialize_simulation()
        
        for _ in range(max_steps):
            if self.step >= max_steps:
                break
            self.run_simulation_step()
            time.sleep(0.05)
    
    def print_event_log(self):
        print("\n" + "="*60)
        print("COMPLETE EVENT LOG")
        print("="*60)
        for event in self.event_log[-30:]:
            print(event)


def main():
    grid = create_demand_grid()
    
    if not grid.get_medical_pickup_locations():
        grid.set_cell(2, 2, is_medical_pickup=True)
    
    simulator = DeliverySimulator(grid, budget=10000)
    simulator.run_full_simulation(max_steps=20)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()