import numpy as np
from scipy import stats
import pandas as pd


class ReorderPointCalculator:
    SERVICE_LEVEL_Z = {
        0.50: 0.00,
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.975: 1.96,
        0.99: 2.33,
        0.995: 2.58,
        0.999: 3.09
    }
    
    def __init__(self, 
                 daily_demand_mean, 
                 daily_demand_std, 
                 lead_time_days, 
                 service_level=0.95,
                 lead_time_std=0):
      
        self.demand_mean = daily_demand_mean
        self.demand_std = daily_demand_std
        self.lead_time_mean = lead_time_days
        self.lead_time_std = lead_time_std
        self.service_level = service_level
        
        # Get Z-score for service level
        self.z_score = self._get_z_score(service_level)
    
    def _get_z_score(self, service_level):
        """Get Z-score for given service level"""
        if service_level in self.SERVICE_LEVEL_Z:
            return self.SERVICE_LEVEL_Z[service_level]
        else:
            return stats.norm.ppf(service_level)
    
    def calculate_safety_stock(self, method='simple'):
        """
        Calculate safety stock
        
        Args:
            method: 'simple' (fixed lead time) or 'advanced' (variable lead time)
        
        Returns:
            float: Safety stock quantity
        """
        if method == 'simple':
            safety_stock = self.z_score * self.demand_std * np.sqrt(self.lead_time_mean)
        
        elif method == 'advanced':
            variance = (
                self.lead_time_mean * (self.demand_std ** 2) +
                (self.demand_mean ** 2) * (self.lead_time_std ** 2)
            )
            safety_stock = self.z_score * np.sqrt(variance)
        
        else:
            raise ValueError("Method must be 'simple' or 'advanced'")
        
        return round(safety_stock, 0)
    
    def calculate_reorder_point(self, method='simple'):
        """
        Calculate reorder point
        
        Args:
            method: 'simple' or 'advanced'
        
        Returns:
            float: Reorder point quantity
        """
        expected_demand = self.demand_mean * self.lead_time_mean
        
        safety_stock = self.calculate_safety_stock(method)
        
        rop = expected_demand + safety_stock
        
        return round(rop, 0)
    
    def get_inventory_policy(self, method='simple'):
        """
        Get complete inventory policy
        
        Args:
            method: 'simple' or 'advanced'
        
        Returns:
            dict: Complete inventory policy
        """
        rop = self.calculate_reorder_point(method)
        safety_stock = self.calculate_safety_stock(method)
        expected_demand = self.demand_mean * self.lead_time_mean
        
        stockout_prob = 1 - self.service_level
        
        orders_per_year = (self.demand_mean * 365) / rop
        expected_stockouts_per_year = orders_per_year * stockout_prob
        
        return {
            'reorder_point': rop,
            'safety_stock': safety_stock,
            'expected_demand_during_lead_time': round(expected_demand, 0),
            'service_level': f"{self.service_level * 100:.1f}%",
            'stockout_probability': f"{stockout_prob * 100:.2f}%",
            'z_score': round(self.z_score, 2),
            'avg_daily_demand': round(self.demand_mean, 2),
            'demand_volatility': round(self.demand_std, 2),
            'coefficient_of_variation': round((self.demand_std / self.demand_mean) * 100, 1) if self.demand_mean > 0 else 0,
            'lead_time_days': self.lead_time_mean,
            'expected_stockouts_per_year': round(expected_stockouts_per_year, 2)
        }
    
    def service_level_analysis(self, service_levels=[0.85, 0.90, 0.95, 0.975, 0.99]):
       
        results = []
        
        for sl in service_levels:
            calc = ReorderPointCalculator(
                self.demand_mean,
                self.demand_std,
                self.lead_time_mean,
                sl,
                self.lead_time_std
            )
            
            policy = calc.get_inventory_policy()
            
            results.append({
                'service_level': f"{sl*100:.1f}%",
                'z_score': policy['z_score'],
                'safety_stock': policy['safety_stock'],
                'reorder_point': policy['reorder_point'],
                'stockout_probability': policy['stockout_probability']
            })
        
        return pd.DataFrame(results)
    
    def lead_time_analysis(self, lead_times=[7, 14, 21, 30]):
        
        results = []
        
        for lt in lead_times:
            calc = ReorderPointCalculator(
                self.demand_mean,
                self.demand_std,
                lt,
                self.service_level,
                self.lead_time_std
            )
            
            policy = calc.get_inventory_policy()
            
            results.append({
                'lead_time_days': lt,
                'expected_demand': policy['expected_demand_during_lead_time'],
                'safety_stock': policy['safety_stock'],
                'reorder_point': policy['reorder_point']
            })
        
        return pd.DataFrame(results)


def calculate_rop_for_product(product_sales, product_info, service_level=0.95):
    
    daily_demand_mean = product_sales['quantity_sold'].mean()
    daily_demand_std = product_sales['quantity_sold'].std()
    
    lead_time = product_info['lead_time_days']
    
    calc = ReorderPointCalculator(
        daily_demand_mean=daily_demand_mean,
        daily_demand_std=daily_demand_std,
        lead_time_days=lead_time,
        service_level=service_level
    )
    
    policy = calc.get_inventory_policy(method='simple')
    
    policy['product_id'] = product_info['product_id']
    policy['category'] = product_info['category']
    
    return policy


class DynamicReorderPoint:
   
    
    def __init__(self, forecast_demand, lead_time, service_level=0.95):
       
        self.forecast = np.array(forecast_demand)
        self.lead_time = lead_time
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)
    
    def calculate_dynamic_rop(self, forecast_horizon=None):
        
        if forecast_horizon is None:
            forecast_horizon = self.lead_time
        
        forecast_period = self.forecast[:forecast_horizon]
        
        expected_demand = forecast_period.sum()
        
        forecast_std = np.std(forecast_period)
        
        safety_stock = self.z_score * forecast_std * np.sqrt(self.lead_time)
        
        rop = expected_demand + safety_stock
        
        return {
            'reorder_point': round(rop, 0),
            'expected_demand': round(expected_demand, 0),
            'safety_stock': round(safety_stock, 0),
            'forecast_mean': round(forecast_period.mean(), 2),
            'forecast_std': round(forecast_std, 2),
            'service_level': f"{self.service_level * 100}%"
        }


def main():
    """Example usage"""
    print("="*60)
    print("REORDER POINT (ROP) & SAFETY STOCK CALCULATOR")
    print("="*60)
    
    daily_demand_mean = 50  
    daily_demand_std = 15  
    lead_time = 14  
    service_level = 0.95  
    
    print(f"\nProduct Parameters:")
    print(f"Average Daily Demand: {daily_demand_mean} units")
    print(f"Demand Std Dev: {daily_demand_std} units")
    print(f"Lead Time: {lead_time} days")
    print(f"Target Service Level: {service_level*100}%")
    
    calc = ReorderPointCalculator(
        daily_demand_mean,
        daily_demand_std,
        lead_time,
        service_level
    )
    
    policy = calc.get_inventory_policy()
    
    print(f"\n{'='*60}")
    print("INVENTORY POLICY")
    print(f"{'='*60}")
    for key, value in policy.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\n{'='*60}")
    print("SERVICE LEVEL ANALYSIS")
    print(f"{'='*60}")
    sl_analysis = calc.service_level_analysis()
    print(sl_analysis.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("LEAD TIME SENSITIVITY")
    print(f"{'='*60}")
    lt_analysis = calc.lead_time_analysis()
    print(lt_analysis.to_string(index=False))


if __name__ == "__main__":
    main()
