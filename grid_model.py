

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ZoneType(Enum):
    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"
    INDUSTRIAL = "Industrial"
    HOSPITAL = "Hospital"
    SCHOOL = "School"
    OPEN_FIELD = "Open Field"


class DensityLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class GridCell:
    row: int
    col: int
    zone: ZoneType = ZoneType.OPEN_FIELD
    density: int = 0
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0
    
    def __post_init__(self):
        if not (0 <= self.row <= 9):
            raise ValueError(f"Row {self.row} must be between 0 and 9")
        if not (0 <= self.col <= 9):
            raise ValueError(f"Col {self.col} must be between 0 and 9")
    
    def to_dict(self) -> Dict:
        return {
            'row': self.row,
            'col': self.col,
            'zone': self.zone.value,
            'density': self.density,
            'is_hub': self.is_hub,
            'is_charging': self.is_charging,
            'is_medical_pickup': self.is_medical_pickup,
            'no_fly': self.no_fly,
            'demand': self.demand
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GridCell':
        return cls(
            row=data['row'],
            col=data['col'],
            zone=ZoneType(data['zone']),
            density=data['density'],
            is_hub=data['is_hub'],
            is_charging=data['is_charging'],
            is_medical_pickup=data['is_medical_pickup'],
            no_fly=data['no_fly'],
            demand=data['demand']
        )


@dataclass
class Delivery:
    delivery_id: int
    pickup_location: Tuple[int, int]
    dropoff_location: Tuple[int, int]
    weight_kg: float
    priority: int = 2
    assigned_drone_id: Optional[str] = None
    status: str = "pending"
    
    def __post_init__(self):
        if not (1 <= self.priority <= 3):
            raise ValueError(f"Priority {self.priority} must be between 1 and 3")


class DroneType(Enum):
    LIGHT = "Light"
    HEAVY = "Heavy"


@dataclass
class Drone:
    drone_id: str
    drone_type: DroneType
    current_location: Tuple[int, int]
    battery_level: float = 100.0
    assigned_delivery: Optional[Delivery] = None
    route: List[Tuple[int, int]] = field(default_factory=list)
    route_index: int = 0
    status: str = "idle"
    is_returning_to_hub: bool = False
    
    @property
    def cost(self) -> int:
        return 1000 if self.drone_type == DroneType.LIGHT else 1800
    
    @property
    def payload_capacity(self) -> int:
        return 2 if self.drone_type == DroneType.LIGHT else 5
    
    @property
    def range_cells(self) -> int:
        return 12 if self.drone_type == DroneType.LIGHT else 20
    
    def can_carry(self, weight_kg: float) -> bool:
        return weight_kg <= self.payload_capacity
    
    def has_battery_for_route(self, route_length: int) -> bool:
        estimated_consumption = route_length * 2
        return self.battery_level >= estimated_consumption + 10


class AeroNetGrid:
    
    def __init__(self, size: int = 10):
        self.size = size
        self.grid: List[List[GridCell]] = []
        self.drones: Dict[str, Drone] = {}
        self.deliveries: List[Delivery] = []
        self.completed_deliveries: List[Delivery] = []
        self.failed_deliveries: List[Delivery] = []
        self.delayed_deliveries: List[Delivery] = []
        self.event_log: List[str] = []
        self.simulation_step: int = 0
        
        self._initialize_empty_grid()
    
    def _initialize_empty_grid(self) -> None:
        self.grid = []
        for row in range(self.size):
            row_cells = []
            for col in range(self.size):
                row_cells.append(GridCell(row=row, col=col))
            self.grid.append(row_cells)
    
    def get_cell(self, row: int, col: int) -> Optional[GridCell]:
        if 0 <= row < self.size and 0 <= col < self.size:
            return self.grid[row][col]
        return None
    
    def set_cell(self, row: int, col: int, **kwargs) -> bool:
        cell = self.get_cell(row, col)
        if cell:
            for key, value in kwargs.items():
                if hasattr(cell, key):
                    if key == 'zone' and isinstance(value, str):
                        setattr(cell, key, ZoneType(value))
                    else:
                        setattr(cell, key, value)
            return True
        return False
    
    def set_no_fly(self, row: int, col: int, value: bool) -> bool:
        return self.set_cell(row, col, no_fly=value)
    
    def get_hub_locations(self) -> List[Tuple[int, int]]:
        hubs = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col].is_hub:
                    hubs.append((row, col))
        return hubs
    
    def get_charging_locations(self) -> List[Tuple[int, int]]:
        chargers = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col].is_charging:
                    chargers.append((row, col))
        return chargers
    
    def get_medical_pickup_locations(self) -> List[Tuple[int, int]]:
        medical = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col].is_medical_pickup:
                    medical.append((row, col))
        return medical
    
    def get_cells_by_zone(self, zone_type: ZoneType) -> List[Tuple[int, int]]:
        cells = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col].zone == zone_type:
                    cells.append((row, col))
        return cells
    
    def is_cell_blocked(self, row: int, col: int) -> bool:
        cell = self.get_cell(row, col)
        return cell is not None and cell.no_fly
    
    def add_drone(self, drone: Drone) -> None:
        self.drones[drone.drone_id] = drone
        self.log_event(f"Drone {drone.drone_id} ({drone.drone_type.value}) added at {drone.current_location}")
    
    def add_delivery(self, delivery: Delivery) -> None:
        self.deliveries.append(delivery)
        self.log_event(f"Delivery {delivery.delivery_id} added: {delivery.pickup_location} -> {delivery.dropoff_location}")
    
    def log_event(self, message: str) -> None:
        self.event_log.append(f"Step {self.simulation_step}: {message}")
        print(f"[Step {self.simulation_step}] {message}")
    
    def get_summary(self) -> Dict:
        return {
            'step': self.simulation_step,
            'drones': {
                'total': len(self.drones),
                'idle': sum(1 for d in self.drones.values() if d.status == 'idle'),
                'moving': sum(1 for d in self.drones.values() if d.status == 'moving'),
                'disrupted': sum(1 for d in self.drones.values() if d.status == 'disrupted')
            },
            'deliveries': {
                'pending': len(self.deliveries),
                'completed': len(self.completed_deliveries),
                'failed': len(self.failed_deliveries),
                'delayed': len(self.delayed_deliveries)
            },
            'hub_count': len(self.get_hub_locations()),
            'charging_stations': len(self.get_charging_locations())
        }
    
    def print_summary(self) -> None:
        summary = self.get_summary()
        print("\n" + "="*50)
        print("AERONET LITE SIMULATION SUMMARY")
        print("="*50)
        print(f"Step: {summary['step']}")
        print(f"\nDrones: {summary['drones']['total']} total")
        print(f"  - Idle: {summary['drones']['idle']}")
        print(f"  - Moving: {summary['drones']['moving']}")
        print(f"  - Disrupted: {summary['drones']['disrupted']}")
        print(f"\nDeliveries:")
        print(f"  - Completed: {summary['deliveries']['completed']}")
        print(f"  - Delayed: {summary['deliveries']['delayed']}")
        print(f"  - Failed: {summary['deliveries']['failed']}")
        print(f"  - Pending: {summary['deliveries']['pending']}")
        print(f"\nInfrastructure:")
        print(f"  - Drone Hubs: {summary['hub_count']}")
        print(f"  - Charging Stations: {summary['charging_stations']}")
        print("="*50)


def create_sample_grid() -> AeroNetGrid:
    grid = AeroNetGrid(size=10)
    
    for row in range(10):
        for col in range(10):
            if row < 4 and col < 5:
                grid.set_cell(row, col, zone=ZoneType.RESIDENTIAL, density=5000)
            elif 3 <= row <= 6 and 3 <= col <= 7:
                grid.set_cell(row, col, zone=ZoneType.COMMERCIAL, density=3000)
            elif row > 6 and col > 6:
                grid.set_cell(row, col, zone=ZoneType.INDUSTRIAL, density=1000)
            elif (row, col) in [(2, 2), (2, 7), (7, 2)]:
                grid.set_cell(row, col, zone=ZoneType.HOSPITAL, density=2000)
            elif (row, col) in [(1, 8), (5, 1), (8, 5)]:
                grid.set_cell(row, col, zone=ZoneType.SCHOOL, density=4000)
            else:
                grid.set_cell(row, col, zone=ZoneType.OPEN_FIELD, density=500)
    
    grid.set_cell(0, 0, is_hub=True, is_charging=True)
    grid.set_cell(9, 9, is_hub=True, is_charging=True)
    grid.set_cell(5, 5, is_hub=True)
    
    grid.set_cell(2, 2, is_medical_pickup=True)
    grid.set_cell(2, 7, is_medical_pickup=True)
    
    for row in range(10):
        for col in range(10):
            cell = grid.get_cell(row, col)
            if cell:
                if cell.zone == ZoneType.RESIDENTIAL:
                    cell.demand = 50 + (row * 10) + (col * 5)
                elif cell.zone == ZoneType.COMMERCIAL:
                    cell.demand = 80 + (row * 15)
                else:
                    cell.demand = 10
    
    return grid


if __name__ == "__main__":
    print("Testing AeroNet Grid Model...")
    
    grid = create_sample_grid()
    
    print(f"\nGrid size: {grid.size}x{grid.size}")
    print(f"Hub locations: {grid.get_hub_locations()}")
    print(f"Charging locations: {grid.get_charging_locations()}")
    print(f"Medical pickup locations: {grid.get_medical_pickup_locations()}")
    
    cell = grid.get_cell(0, 0)
    if cell:
        print(f"\nCell (0,0): Zone={cell.zone.value}, Hub={cell.is_hub}, Charging={cell.is_charging}")
    
    from datetime import datetime
    drone = Drone(
        drone_id="D001",
        drone_type=DroneType.LIGHT,
        current_location=(0, 0),
        battery_level=85.0
    )
    grid.add_drone(drone)
    
    delivery = Delivery(
        delivery_id=1,
        pickup_location=(0, 0),
        dropoff_location=(3, 4),
        weight_kg=1.5,
        priority=1
    )
    grid.add_delivery(delivery)
    
    grid.print_summary()
    
    print("\nGrid model ready!")