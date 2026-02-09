import numpy as np
import pandas as pd


class EOQCalculator:
    """
    Calculate Economic Order Quantity and related inventory metrics
    """
    
    def __init__(self, annual_demand, order_cost, holding_cost_percent, unit_cost):
        """
        Initialize EOQ calculator
        
        Args:
            annual_demand: Total annual demand in units
            order_cost: Fixed cost per order ($)
            holding_cost_percent: Annual holding cost as percentage of unit cost (e.g., 0.25 for 25%)
            unit_cost: Cost per unit ($)
        """
        self.annual_demand = annual_demand
        self.order_cost = order_cost
        self.unit_cost = unit_cost
        self.holding_cost_percent = holding_cost_percent
        self.holding_cost = unit_cost * holding_cost_percent
    
    def calculate_eoq(self):
        """
        Calculate Economic Order Quantity
        
        Returns:
            float: Optimal order quantity
        """
        if self.holding_cost == 0:
            raise ValueError("Holding cost cannot be zero")
        
        eoq = np.sqrt((2 * self.annual_demand * self.order_cost) / self.holding_cost)
        return round(eoq, 0)
    
    def calculate_total_cost(self, order_quantity):
        """
        Calculate total annual inventory cost for given order quantity
        
        Args:
            order_quantity: Order quantity to evaluate
        
        Returns:
            dict: Breakdown of costs
        """
        if order_quantity == 0:
            raise ValueError("Order quantity cannot be zero")
        
        n_orders = self.annual_demand / order_quantity
        
        ordering_cost = n_orders * self.order_cost
        
        holding_cost_total = (order_quantity / 2) * self.holding_cost
        
        purchase_cost = self.annual_demand * self.unit_cost
        
        total_cost = ordering_cost + holding_cost_total + purchase_cost
        
        return {
            'ordering_cost': round(ordering_cost, 2),
            'holding_cost': round(holding_cost_total, 2),
            'purchase_cost': round(purchase_cost, 2),
            'total_cost': round(total_cost, 2),
            'variable_cost': round(ordering_cost + holding_cost_total, 2),
            'n_orders': round(n_orders, 2)
        }
    
    def get_optimal_policy(self):
        """
        Get complete optimal ordering policy
        
        Returns:
            dict: Complete EOQ-based policy
        """
        eoq = self.calculate_eoq()
        costs = self.calculate_total_cost(eoq)
        
        n_orders = costs['n_orders']
        days_between_orders = 365 / n_orders
        
        if n_orders >= 52:
            order_frequency = f"{n_orders / 52:.1f}x per week"
        elif n_orders >= 12:
            order_frequency = f"{n_orders / 12:.1f}x per month"
        else:
            order_frequency = f"{n_orders:.1f}x per year"
        
        return {
            'eoq': eoq,
            'orders_per_year': costs['n_orders'],
            'days_between_orders': round(days_between_orders, 1),
            'order_frequency': order_frequency,
            'annual_ordering_cost': costs['ordering_cost'],
            'annual_holding_cost': costs['holding_cost'],
            'annual_purchase_cost': costs['purchase_cost'],
            'total_annual_cost': costs['total_cost'],
            'total_variable_cost': costs['variable_cost'],
            'average_inventory': round(eoq / 2, 0)
        }
    
    def sensitivity_analysis(self, demand_range=(-30, 30), step=10):
        """
        Perform sensitivity analysis on demand changes
        
        Args:
            demand_range: Tuple of (min_change_percent, max_change_percent)
            step: Step size for analysis
        
        Returns:
            DataFrame: Sensitivity analysis results
        """
        results = []
        
        base_demand = self.annual_demand
        
        for change_pct in range(demand_range[0], demand_range[1] + 1, step):
            new_demand = base_demand * (1 + change_pct / 100)
            
            calc = EOQCalculator(
                new_demand,
                self.order_cost,
                self.holding_cost_percent,
                self.unit_cost
            )
            
            policy = calc.get_optimal_policy()
            
            results.append({
                'demand_change_pct': change_pct,
                'new_annual_demand': int(new_demand),
                'eoq': policy['eoq'],
                'orders_per_year': policy['orders_per_year'],
                'total_cost': policy['total_annual_cost'],
                'total_variable_cost': policy['total_variable_cost']
            })
        
        return pd.DataFrame(results)
    
    def compare_order_quantities(self, quantities):
        """
        Compare costs for different order quantities
        
        Args:
            quantities: List of order quantities to compare
        
        Returns:
            DataFrame: Comparison results
        """
        results = []
        
        for qty in quantities:
            costs = self.calculate_total_cost(qty)
            
            results.append({
                'order_quantity': qty,
                **costs
            })
        
        return pd.DataFrame(results)


def calculate_eoq_for_product(product_sales, product_info, 
                               order_cost=50, 
                               holding_cost_percent=0.25):
    """
    Calculate EOQ for a specific product
    
    Args:
        product_sales: DataFrame with historical sales for product
        product_info: Series with product information
        order_cost: Fixed cost per order
        holding_cost_percent: Annual holding cost percentage
    
    Returns:
        dict: EOQ policy for product
    """
    daily_avg = product_sales['quantity_sold'].mean()
    annual_demand = daily_avg * 365
    
    unit_cost = product_info['cost']
    
    calc = EOQCalculator(
        annual_demand=annual_demand,
        order_cost=order_cost,
        holding_cost_percent=holding_cost_percent,
        unit_cost=unit_cost
    )
    
    policy = calc.get_optimal_policy()
    
    policy['product_id'] = product_info['product_id']
    policy['category'] = product_info['category']
    policy['unit_cost'] = unit_cost
    policy['avg_daily_demand'] = round(daily_avg, 2)
    
    return policy


def main():
    """Example usage"""
    print("="*60)
    print("ECONOMIC ORDER QUANTITY (EOQ) CALCULATOR")
    print("="*60)
    
    annual_demand = 12000  
    order_cost = 50  
    unit_cost = 25  
    holding_cost_percent = 0.25  
    
    print(f"\nProduct Parameters:")
    print(f"Annual Demand: {annual_demand:,} units")
    print(f"Order Cost: ${order_cost}")
    print(f"Unit Cost: ${unit_cost}")
    print(f"Holding Cost: {holding_cost_percent*100}% of unit cost per year")
    
    calc = EOQCalculator(annual_demand, order_cost, holding_cost_percent, unit_cost)
    policy = calc.get_optimal_policy()
    
    print(f"\n{'='*60}")
    print("OPTIMAL ORDERING POLICY")
    print(f"{'='*60}")
    for key, value in policy.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Sensitivity analysis
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    sensitivity = calc.sensitivity_analysis(demand_range=(-30, 30), step=10)
    print(sensitivity.to_string(index=False))
    
    # Compare order quantities
    print(f"\n{'='*60}")
    print("ORDER QUANTITY COMPARISON")
    print(f"{'='*60}")
    eoq = policy['eoq']
    quantities = [eoq * 0.5, eoq * 0.75, eoq, eoq * 1.25, eoq * 1.5]
    comparison = calc.compare_order_quantities(quantities)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
