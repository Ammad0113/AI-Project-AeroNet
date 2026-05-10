
import heapq
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from grid_model import AeroNetGrid, ZoneType


@dataclass
class Node:
    position: Tuple[int, int]
    g_cost: float = float('inf')
    h_cost: float = float('inf')
    parent: Optional['Node'] = None
    
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class AStarPlanner:
    
    NORMAL_COST = 1.0
    COMMERCIAL_CORRIDOR_COST = 0.8
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, grid: AeroNetGrid):
        self.grid = grid
        
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_movement_cost(self, row: int, col: int) -> float:
        cell = self.grid.get_cell(row, col)
        if cell and cell.zone == ZoneType.COMMERCIAL:
            return self.COMMERCIAL_CORRIDOR_COST
        return self.NORMAL_COST
    
    def is_valid_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.grid.size and 0 <= col < self.grid.size):
            return False
        cell = self.grid.get_cell(row, col)
        if cell and cell.no_fly:
            return False
        return True
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        row, col = position
        neighbors = []
        
        for dr, dc in self.DIRECTIONS:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_cell(new_row, new_col):
                cost = self.get_movement_cost(new_row, new_col)
                neighbors.append((new_row, new_col, cost))
        
        return neighbors
    
    def reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        path = []
        current = goal_node
        while current:
            path.append(current.position)
            current = current.parent
        return list(reversed(path))
    
    def find_path(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        verbose: bool = False
    ) -> Tuple[Optional[List[Tuple[int, int]]], float, bool]:
        
        if not self.is_valid_cell(start[0], start[1]):
            if verbose:
                print(f"Start cell {start} is invalid or no-fly zone")
            return None, float('inf'), False
        
        if not self.is_valid_cell(goal[0], goal[1]):
            if verbose:
                print(f"Goal cell {goal} is invalid or no-fly zone")
            return None, float('inf'), False
        
        open_set = []
        open_dict: Dict[Tuple[int, int], Node] = {}
        closed_set: Set[Tuple[int, int]] = set()
        
        start_node = Node(
            position=start,
            g_cost=0,
            h_cost=self.manhattan_distance(start, goal)
        )
        
        heapq.heappush(open_set, start_node)
        open_dict[start] = start_node
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            
            if current.position in open_dict:
                del open_dict[current.position]
            
            if current.position == goal:
                path = self.reconstruct_path(current)
                if verbose:
                    print(f"Path found! Length: {len(path)} cells, Cost: {current.g_cost:.2f}")
                    print(f"Nodes explored: {nodes_explored}")
                return path, current.g_cost, True
            
            closed_set.add(current.position)
            
            for neighbor_row, neighbor_col, move_cost in self.get_neighbors(current.position):
                neighbor_pos = (neighbor_row, neighbor_col)
                
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g = current.g_cost + move_cost
                neighbor_node = open_dict.get(neighbor_pos)
                
                if neighbor_node is None:
                    neighbor_node = Node(
                        position=neighbor_pos,
                        g_cost=tentative_g,
                        h_cost=self.manhattan_distance(neighbor_pos, goal),
                        parent=current
                    )
                    open_dict[neighbor_pos] = neighbor_node
                    heapq.heappush(open_set, neighbor_node)
                    nodes_explored += 1
                    
                elif tentative_g < neighbor_node.g_cost:
                    neighbor_node.g_cost = tentative_g
                    neighbor_node.parent = current
                    heapq.heappush(open_set, neighbor_node)
        
        if verbose:
            print(f"No path found from {start} to {goal}")
            print(f"Nodes explored: {nodes_explored}")
        
        return None, float('inf'), False
    
    def plan_delivery_route(
        self,
        hub: Tuple[int, int],
        pickup: Tuple[int, int],
        dropoff: Tuple[int, int],
        verbose: bool = False
    ) -> Tuple[Optional[List[Tuple[int, int]]], float, bool]:
        
        if verbose:
            print(f"Planning route: Hub {hub} -> Pickup {pickup} -> Dropoff {dropoff} -> Hub {hub}")
        
        path1, cost1, success1 = self.find_path(hub, pickup, verbose)
        if not success1:
            if verbose:
                print(f"Failed to find path from hub to pickup")
            return None, float('inf'), False
        
        path2, cost2, success2 = self.find_path(pickup, dropoff, verbose)
        if not success2:
            if verbose:
                print(f"Failed to find path from pickup to dropoff")
            return None, float('inf'), False
        
        path3, cost3, success3 = self.find_path(dropoff, hub, verbose)
        if not success3:
            if verbose:
                print(f"Failed to find path from dropoff to hub")
            return None, float('inf'), False
        
        full_route = path1 + path2[1:] + path3[1:]
        total_cost = cost1 + cost2 + cost3
        
        if verbose:
            print(f"Complete route planned! Total cells: {len(full_route)}, Total cost: {total_cost:.2f}")
        
        return full_route, total_cost, True
    
    def find_alternative_path(
        self,
        current_pos: Tuple[int, int],
        target: Tuple[int, int],
        blocked_cells: Set[Tuple[int, int]] = None,
        verbose: bool = False
    ) -> Tuple[Optional[List[Tuple[int, int]]], float, bool]:
        
        temp_no_fly = []
        
        if blocked_cells:
            for row, col in blocked_cells:
                cell = self.grid.get_cell(row, col)
                if cell and not cell.no_fly:
                    cell.no_fly = True
                    temp_no_fly.append((row, col))
        
        path, cost, success = self.find_path(current_pos, target, verbose)
        
        for row, col in temp_no_fly:
            cell = self.grid.get_cell(row, col)
            if cell:
                cell.no_fly = False
        
        return path, cost, success


def demo_astar():
    from grid_model import create_sample_grid
    
    print("\n" + "="*60)
    print("A* PATH PLANNER DEMONSTRATION")
    print("="*60)
    
    grid = create_sample_grid()
    
    grid.set_no_fly(3, 3, True)
    grid.set_no_fly(3, 4, True)
    grid.set_no_fly(4, 3, True)
    grid.set_no_fly(5, 5, True)
    
    planner = AStarPlanner(grid)
    
    print("\nTEST 1: Simple path (no obstacles)")
    start = (0, 0)
    goal = (0, 5)
    path, cost, success = planner.find_path(start, goal, verbose=True)
    
    if success and path:
        print(f"Path length: {len(path)} cells")
    
    print("\nTEST 2: Path with no-fly zones")
    start = (0, 0)
    goal = (9, 9)
    path, cost, success = planner.find_path(start, goal, verbose=True)
    
    if success and path:
        print(f"Path length: {len(path)} cells, Total cost: {cost:.2f}")
    
    print("\nTEST 3: Complete delivery route")
    hub = (0, 0)
    pickup = (2, 5)
    dropoff = (7, 8)
    
    full_route, total_cost, success = planner.plan_delivery_route(
        hub, pickup, dropoff, verbose=True
    )
    
    print("\nTEST 4: Impossible path (completely blocked)")
    grid.set_no_fly(0, 1, True)
    grid.set_no_fly(1, 0, True)
    
    path, cost, success = planner.find_path((0, 0), (9, 9), verbose=True)


if __name__ == "__main__":
    demo_astar()
    print("\n" + "="*60)
    print("A* Path Planner ready for integration")
    print("="*60)