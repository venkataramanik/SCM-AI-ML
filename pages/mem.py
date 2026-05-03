import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class EchelonNode:
    """Represents a node in multi-echelon supply chain"""
    name: str
    holding_cost: float
    shortage_cost: float
    lead_time: int
    order_cost: float
    demand_mean: float
    demand_std: float
    service_level: float
    capacity: float = float('inf')

class NewsvendorOptimizer:
    """Classic Newsvendor problem with extensions"""
    
    def __init__(self, selling_price: float, cost: float, salvage: float):
        self.p = selling_price
        self.c = cost
        self.s = salvage
        self.cu = selling_price - cost  # underage cost
        self.co = cost - salvage        # overage cost
        
    def optimal_order_quantity(self, demand_mean: float, demand_std: float, 
                               distribution: str = 'normal') -> float:
        """Calculate optimal order quantity using critical ratio"""
        critical_ratio = self.cu / (self.cu + self.co)
        
        if distribution == 'normal':
            z_score = stats.norm.ppf(critical_ratio)
            return demand_mean + z_score * demand_std
        elif distribution == 'lognormal':
            # Log-normal distribution parameters
            mu = np.log(demand_mean**2 / np.sqrt(demand_std**2 + demand_mean**2))
            sigma = np.sqrt(np.log(1 + (demand_std**2 / demand_mean**2)))
            return stats.lognorm.ppf(critical_ratio, s=sigma, scale=np.exp(mu))
        
        return demand_mean
    
    def expected_profit(self, order_quantity: float, demand_mean: float, 
                       demand_std: float, simulations: int = 10000) -> Dict:
        """Monte Carlo simulation for expected profit"""
        demands = np.random.normal(demand_mean, demand_std, simulations)
        demands = np.maximum(demands, 0)  # Non-negative demand
        
        sales = np.minimum(order_quantity, demands)
        leftover = np.maximum(order_quantity - demands, 0)
        shortage = np.maximum(demands - order_quantity, 0)
        
        revenue = sales * self.p
        salvage_revenue = leftover * self.s
        total_cost = order_quantity * self.c
        
        profit = revenue + salvage_revenue - total_cost
        
        return {
            'expected_profit': np.mean(profit),
            'profit_std': np.std(profit),
            'service_level': np.mean(sales / demands),
            'expected_shortage': np.mean(shortage),
            'expected_leftover': np.mean(leftover),
            'profit_percentiles': {
                '5%': np.percentile(profit, 5),
                '25%': np.percentile(profit, 25),
                '50%': np.percentile(profit, 50),
                '75%': np.percentile(profit, 75),
                '95%': np.percentile(profit, 95)
            }
        }

class MultiEchelonInventory:
    """Multi-echelon inventory optimization using decomposition"""
    
    def __init__(self, nodes: List[EchelonNode]):
        self.nodes = nodes
        self.n_echelons = len(nodes)
        
    def calculate_echelon_costs(self) -> List[Dict]:
        """Calculate echelon-based costs and optimal policies"""
        results = []
        
        for i, node in enumerate(self.nodes):
            # Echelon holding cost (value added at this stage)
            echelon_holding = node.holding_cost
            if i < self.n_echelons - 1:
                echelon_holding -= self.nodes[i+1].holding_cost
            
            # Safety stock calculation
            z_score = stats.norm.ppf(node.service_level)
            lead_time_demand_std = node.demand_std * np.sqrt(node.lead_time)
            safety_stock = z_score * lead_time_demand_std
            
            # Reorder point
            lead_time_demand = node.demand_mean * node.lead_time
            reorder_point = lead_time_demand + safety_stock
            
            # Economic Order Quantity
            eoq = np.sqrt(2 * node.order_cost * node.demand_mean / node.holding_cost)
            
            # Order-up-to level
            order_up_to = reorder_point + eoq
            
            results.append({
                'node': node.name,
                'echelon_holding_cost': echelon_holding,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'eoq': eoq,
                'order_up_to_level': order_up_to,
                'avg_inventory': safety_stock + eoq/2,
                'total_cost': (node.holding_cost * (safety_stock + eoq/2) + 
                             node.order_cost * (node.demand_mean / eoq))
            })
            
        return results
    
    def optimize_inventory(self, method: str = 'decomposition') -> Dict:
        """Main optimization routine"""
        if method == 'decomposition':
            return self.calculate_echelon_costs()
        
        # Could implement more sophisticated methods:
        # - Guaranteed Service Model (GSM)
        # - Stochastic Service Model (SSM)
        # - Multi-objective optimization
        
    def simulate_network(self, policies: List[Dict], 
                        periods: int = 365) -> pd.DataFrame:
        """Simulate the multi-echelon network performance"""
        import pandas as pd
        
        # Initialize inventory levels
        inventory = [policy['order_up_to_level'] for policy in policies]
        backlog = [0] * self.n_echelons
        
        simulation_data = []
        
        for t in range(periods):
            # Generate random demands at each echelon
            demands = [max(0, np.random.normal(node.demand_mean, node.demand_std))
                      for node in self.nodes]
            
            # Fulfillment logic
            fulfilled = [0] * self.n_echelons
            for i in reversed(range(self.n_echelons)):
                available = inventory[i] + (fulfilled[i+1] if i < self.n_echelons-1 else 0)
                fulfilled[i] = min(demands[i] + backlog[i], available)
                backlog[i] = max(0, demands[i] + backlog[i] - available)
                inventory[i] = available - fulfilled[i]
            
            # Replenishment logic (simplified)
            for i in range(self.n_echelons):
                if inventory[i] <= policies[i]['reorder_point']:
                    inventory[i] = policies[i]['order_up_to_level']
            
            simulation_data.append({
                'period': t,
                **{f'{self.nodes[i].name}_inventory': inventory[i] 
                   for i in range(self.n_echelons)},
                **{f'{self.nodes[i].name}_demand': demands[i] 
                   for i in range(self.n_echelons)},
                **{f'{self.nodes[i].name}_backlog': backlog[i] 
                   for i in range(self.n_echelons)}
            })
        
        return pd.DataFrame(simulation_data)

class SupplyChainNetworkDesign:
    """Facility location and network design optimization using MILP"""
    
    @staticmethod
    def optimize_network(demand_nodes: List[Tuple[float, float, float]],
                        candidate_facilities: List[Tuple[float, float, float, float]],
                        transportation_cost: float = 1.0) -> Dict:
        """
        Optimize facility locations using Mixed Integer Linear Programming
        
        Args:
            demand_nodes: List of (x, y, demand) tuples
            candidate_facilities: List of (x, y, capacity, fixed_cost) tuples
            transportation_cost: Cost per unit distance per unit demand
        """
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value
        import pulp
        
        n_demands = len(demand_nodes)
        n_facilities = len(candidate_facilities)
        
        # Create optimization model
        prob = LpProblem("Facility_Location", LpMinimize)
        
        # Decision variables
        y = [LpVariable(f"open_{j}", cat=LpBinary) for j in range(n_facilities)]
        x = [[LpVariable(f"flow_{j}_{i}", lowBound=0) 
              for i in range(n_demands)] for j in range(n_facilities)]
        
        # Objective: Minimize total cost
        fixed_costs = lpSum(candidate_facilities[j][3] * y[j] 
                           for j in range(n_facilities))
        
        transport_costs = lpSum(
            np.sqrt((candidate_facilities[j][0] - demand_nodes[i][0])**2 + 
                   (candidate_facilities[j][1] - demand_nodes[i][1])**2) *
            transportation_cost * x[j][i]
            for j in range(n_facilities) for i in range(n_demands)
        )
        
        prob += fixed_costs + transport_costs
        
        # Constraints: Meet all demand
        for i in range(n_demands):
            prob += lpSum(x[j][i] for j in range(n_facilities)) == demand_nodes[i][2]
        
        # Constraints: Capacity
        for j in range(n_facilities):
            prob += lpSum(x[j][i] for i in range(n_demands)) <= \
                   candidate_facilities[j][2] * y[j]
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract results
        open_facilities = [j for j in range(n_facilities) if value(y[j]) == 1]
        assignments = {}
        for j in open_facilities:
            for i in range(n_demands):
                if value(x[j][i]) > 0:
                    assignments[(j, i)] = value(x[j][i])
        
        return {
            'status': pulp.LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'open_facilities': open_facilities,
            'num_facilities': len(open_facilities),
            'assignments': assignments,
            'fixed_cost': sum(candidate_facilities[j][3] for j in open_facilities)
        }