

from typing import List, Tuple, Dict, Set
from grid_model import AeroNetGrid, ZoneType, GridCell


class LayoutValidator:
    
    def __init__(self, grid: AeroNetGrid):
        self.grid = grid
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.passed_rules: Set[str] = set()
        self.failed_rules: Set[str] = set()
        
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid.size and 0 <= new_col < self.grid.size:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def check_industrial_safety(self) -> bool:
        print("\n[ R1 ] Checking industrial safety...")
        rule_passed = True
        
        industrial_cells = self.grid.get_cells_by_zone(ZoneType.INDUSTRIAL)
        
        for row, col in industrial_cells:
            for neighbor_row, neighbor_col in self.get_neighbors(row, col):
                neighbor = self.grid.get_cell(neighbor_row, neighbor_col)
                if neighbor and neighbor.zone in [ZoneType.SCHOOL, ZoneType.HOSPITAL]:
                    rule_passed = False
                    self.errors.append({
                        'rule': 'R1',
                        'message': f'Industrial cell ({row}, {col}) is adjacent to {neighbor.zone.value} at ({neighbor_row}, {neighbor_col})',
                        'cells': [(row, col), (neighbor_row, neighbor_col)],
                        'suggestion': f'Move the {neighbor.zone.value} or convert the industrial cell to Open Field'
                    })
        
        if rule_passed:
            self.passed_rules.add('R1')
            print("  R1 PASSED: No industrial cells adjacent to schools or hospitals")
        else:
            self.failed_rules.add('R1')
            print(f"  R1 FAILED: {len(self.errors)} violations found")
        
        return rule_passed
    
    def check_residential_coverage(self) -> bool:
        print("\n[ R2 ] Checking residential coverage...")
        rule_passed = True
        hubs = self.grid.get_hub_locations()
        
        if not hubs:
            rule_passed = False
            self.errors.append({
                'rule': 'R2',
                'message': 'No drone hubs found in the grid!',
                'cells': [],
                'suggestion': 'Add at least one drone hub to the grid'
            })
            self.failed_rules.add('R2')
            print("  R2 FAILED: No drone hubs found")
            return False
        
        residential_cells = self.grid.get_cells_by_zone(ZoneType.RESIDENTIAL)
        
        for row, col in residential_cells:
            min_distance = min(self.manhattan_distance((row, col), hub) for hub in hubs)
            
            if min_distance > 3:
                rule_passed = False
                self.errors.append({
                    'rule': 'R2',
                    'message': f'Residential cell ({row}, {col}) is {min_distance} cells away from nearest hub (max allowed: 3)',
                    'cells': [(row, col)],
                    'suggestion': f'Add a drone hub near ({row}, {col}) or convert this cell to Open Field'
                })
        
        if rule_passed:
            self.passed_rules.add('R2')
            print(f"  R2 PASSED: All {len(residential_cells)} residential cells within 3 cells of a hub")
        else:
            self.failed_rules.add('R2')
            print(f"  R2 FAILED: {len([e for e in self.errors if e['rule'] == 'R2'])} violations found")
        
        return rule_passed
    
    def check_hub_charging_proximity(self) -> bool:
        print("\n[ R3 ] Checking hub-charging proximity...")
        rule_passed = True
        hubs = self.grid.get_hub_locations()
        chargers = self.grid.get_charging_locations()
        
        if not hubs:
            print("  No hubs found, skipping R3 check")
            return True
        
        if not chargers:
            rule_passed = False
            for hub in hubs:
                self.errors.append({
                    'rule': 'R3',
                    'message': f'Hub at {hub} has no charging pads anywhere in grid',
                    'cells': [hub],
                    'suggestion': 'Add charging pads near each drone hub (within 2 cells)'
                })
            self.failed_rules.add('R3')
            print("  R3 FAILED: No charging pads found")
            return False
        
        for hub_row, hub_col in hubs:
            min_distance = min(self.manhattan_distance((hub_row, hub_col), charger) for charger in chargers)
            
            if min_distance > 2:
                rule_passed = False
                self.errors.append({
                    'rule': 'R3',
                    'message': f'Hub at ({hub_row}, {hub_col}) is {min_distance} cells away from nearest charging pad (max allowed: 2)',
                    'cells': [(hub_row, hub_col)],
                    'suggestion': f'Add a charging pad at ({hub_row+1}, {hub_col}) or within 2 cells of this hub'
                })
        
        if rule_passed:
            self.passed_rules.add('R3')
            print(f"  R3 PASSED: All {len(hubs)} hubs have charging pads within 2 cells")
        else:
            self.failed_rules.add('R3')
            print(f"  R3 FAILED: {len([e for e in self.errors if e['rule'] == 'R3'])} violations found")
        
        return rule_passed
    
    def check_medical_access(self) -> bool:
        print("\n[ R4 ] Checking medical access...")
        rule_passed = True
        hospitals = self.grid.get_cells_by_zone(ZoneType.HOSPITAL)
        medical_pickups = self.grid.get_medical_pickup_locations()
        
        if not hospitals:
            self.warnings.append({
                'rule': 'R4',
                'message': 'No hospitals found in grid',
                'cells': [],
                'suggestion': 'Add hospitals for medical delivery simulation'
            })
            print("  R4 WARNING: No hospitals found, skipping check")
            return True
        
        if not medical_pickups:
            rule_passed = False
            self.errors.append({
                'rule': 'R4',
                'message': 'No medical pickup points found in grid',
                'cells': [],
                'suggestion': 'Add medical pickup points near hospitals (within 1 cell)'
            })
            self.failed_rules.add('R4')
            print("  R4 FAILED: No medical pickup points found")
            return False
        
        found_valid = False
        for hospital_row, hospital_col in hospitals:
            for pickup_row, pickup_col in medical_pickups:
                distance = self.manhattan_distance((hospital_row, hospital_col), (pickup_row, pickup_col))
                if distance <= 1:
                    found_valid = True
                    break
            if found_valid:
                break
        
        if not found_valid:
            rule_passed = False
            self.errors.append({
                'rule': 'R4',
                'message': 'No hospital has a medical pickup point within 1 cell',
                'cells': hospitals,
                'suggestion': 'Add medical pickup points adjacent to hospitals'
            })
            self.failed_rules.add('R4')
            print("  R4 FAILED: No hospital-medical pickup adjacency found")
        else:
            self.passed_rules.add('R4')
            print("  R4 PASSED: At least one hospital has a medical pickup within 1 cell")
        
        return rule_passed
    
    def validate_all(self, verbose: bool = True) -> bool:
        if verbose:
            print("\n" + "="*60)
            print("AERONET LITE - LAYOUT VALIDATION")
            print("="*60)
        
        self.errors = []
        self.warnings = []
        self.passed_rules = set()
        self.failed_rules = set()
        
        r1_passed = self.check_industrial_safety()
        r2_passed = self.check_residential_coverage()
        r3_passed = self.check_hub_charging_proximity()
        r4_passed = self.check_medical_access()
        
        all_passed = r1_passed and r2_passed and r3_passed and r4_passed
        
        if verbose:
            self.print_report()
        
        return all_passed
    
    def print_report(self) -> None:
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        total_rules = 4
        passed_count = len(self.passed_rules)
        failed_count = len(self.failed_rules)
        
        print(f"\nSUMMARY: {passed_count}/{total_rules} rules passed, {failed_count} failed")
        
        if failed_count == 0:
            print("\nLAYOUT VALIDITY = True")
            print("All constraints satisfied. Grid is ready for deployment.")
        else:
            print("\nLAYOUT VALIDITY = False")
            print(f"{failed_count} constraint(s) violated. See details below.")
        
        if self.errors:
            print("\n" + "-"*40)
            print("FAILED CONSTRAINTS:")
            print("-"*40)
            
            for i, error in enumerate(self.errors, 1):
                print(f"\n  [{i}] Rule {error['rule']}:")
                print(f"      Issue: {error['message']}")
                if error['cells']:
                    print(f"      Cells affected: {error['cells']}")
                print(f"      Suggestion: {error['suggestion']}")
        
        if self.warnings:
            print("\n" + "-"*40)
            print("WARNINGS:")
            print("-"*40)
            for warning in self.warnings:
                print(f"  - {warning['message']}")
                print(f"    Suggestion: {warning['suggestion']}")
        
        if self.failed_rules:
            print("\n" + "-"*40)
            print("HOW TO FIX THE LAYOUT:")
            print("-"*40)
            
            if 'R1' in self.failed_rules:
                print("  R1: No INDUSTRIAL zone next to SCHOOL or HOSPITAL")
                print("    -> Move the school/hospital or change industrial to OPEN_FIELD")
            
            if 'R2' in self.failed_rules:
                print("  R2: Every RESIDENTIAL cell within 3 cells of a DRONE HUB")
                print("    -> Add more hubs or note which residential cells are too far")
            
            if 'R3' in self.failed_rules:
                print("  R3: Every HUB needs a CHARGING PAD within 2 cells")
                print("    -> Add charging pads adjacent to each hub")
            
            if 'R4' in self.failed_rules:
                print("  R4: At least one HOSPITAL needs a MEDICAL PICKUP within 1 cell")
                print("    -> Add a medical pickup at the hospital or adjacent cell")
        
        print("\n" + "="*60)
    
    def get_suggestions(self) -> List[str]:
        suggestions = []
        for error in self.errors:
            suggestions.append(error['suggestion'])
        return list(set(suggestions))


def create_test_grid_with_violations() -> AeroNetGrid:
    from grid_model import create_sample_grid
    
    grid = create_sample_grid()
    
    grid.set_cell(7, 6, zone=ZoneType.HOSPITAL)
    grid.set_cell(5, 5, is_hub=False)
    grid.set_cell(9, 9, is_hub=False)
    grid.set_cell(9, 9, zone=ZoneType.RESIDENTIAL, density=3000)
    grid.set_cell(9, 9, is_charging=False)
    
    return grid


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING LAYOUT VALIDATOR")
    print("="*60)
    
    print("\nTEST CASE 1: Valid Grid (Should pass all)")
    print("-" * 50)
    from grid_model import create_sample_grid
    good_grid = create_sample_grid()
    validator1 = LayoutValidator(good_grid)
    result1 = validator1.validate_all(verbose=True)
    
    print("\n\nTEST CASE 2: Invalid Grid (Should fail)")
    print("-" * 50)
    bad_grid = create_test_grid_with_violations()
    validator2 = LayoutValidator(bad_grid)
    result2 = validator2.validate_all(verbose=True)
    
    if not result2:
        print("\nAUTO-GENERATED SUGGESTIONS:")
        suggestions = validator2.get_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")