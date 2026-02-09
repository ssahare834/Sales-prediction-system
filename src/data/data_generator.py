
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

class SalesDataGenerator:
    def __init__(self, 
                 n_products=100, 
                 n_years=3, 
                 start_date='2022-01-01',
                 seed=42):
        """
        Initialize data generator
        
        Args:
            n_products: Number of unique products (SKUs)
            n_years: Years of historical data
            start_date: Start date for data generation
            seed: Random seed for reproducibility
        """
        self.n_products = n_products
        self.n_years = n_years
        self.start_date = pd.to_datetime(start_date)
        self.end_date = self.start_date + timedelta(days=365*n_years)
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        
        self.categories = [
            'Electronics', 'Clothing', 'Home & Garden', 
            'Sports', 'Books', 'Toys', 'Food & Beverage',
            'Beauty', 'Automotive', 'Office Supplies'
        ]
        
       
        self.seasonal_products = {
            'winter': ['Heaters', 'Winter Coats', 'Snow Boots'],
            'summer': ['Fans', 'Swimwear', 'Sunscreen'],
            'back_to_school': ['Backpacks', 'Notebooks', 'Pencils'],
            'holiday': ['Decorations', 'Gift Wrap', 'Toys']
        }
        
    def generate_products(self):
        """Generate product master data"""
        print("Generating product data...")
        
        products = []
        for i in range(self.n_products):
            product = {
                'product_id': f'SKU_{i+1:04d}',
                'product_name': f'Product {i+1}',
                'category': np.random.choice(self.categories),
                'subcategory': f'Subcategory {np.random.randint(1, 6)}',
                'price': round(np.random.uniform(5, 500), 2),
                'cost': 0,  # Will calculate based on price
                'supplier': f'Supplier_{np.random.randint(1, 21)}',
                'lead_time_days': np.random.randint(3, 30),
                'weight_kg': round(np.random.uniform(0.1, 20), 2),
                'shelf_life_days': np.random.choice([None, 30, 60, 90, 180, 365]),
                'min_order_quantity': np.random.choice([1, 5, 10, 20, 50]),
                'warehouse_location': np.random.choice(['WH_A', 'WH_B', 'WH_C', 'WH_D']),
            }
            
           
            product['cost'] = round(product['price'] * np.random.uniform(0.4, 0.7), 2)
            
           
            product['margin_percent'] = round(
                ((product['price'] - product['cost']) / product['price']) * 100, 2
            )
            
            products.append(product)
        
        df_products = pd.DataFrame(products)
        
        
        df_products['base_demand'] = np.random.randint(10, 200, self.n_products)
        df_products['seasonality_strength'] = np.random.uniform(0, 1, self.n_products)
        df_products['trend_coefficient'] = np.random.uniform(-0.0005, 0.002, self.n_products)
        df_products['promotion_sensitivity'] = np.random.uniform(1.2, 3.0, self.n_products)
        df_products['day_of_week_pattern'] = np.random.choice(
            ['weekend_high', 'weekday_high', 'uniform'], 
            self.n_products
        )
        
        return df_products
    
    def generate_date_features(self):
        """Generate date range with features"""
        print("Generating date features...")
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        df_dates = pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'day': dates.day,
            'day_of_week': dates.dayofweek,
            'day_name': dates.day_name(),
            'week_of_year': dates.isocalendar().week,
            'quarter': dates.quarter,
            'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
            'is_month_start': dates.is_month_start.astype(int),
            'is_month_end': dates.is_month_end.astype(int),
        })
        
       
        df_dates['is_holiday'] = 0
        df_dates['is_promotion'] = 0
        df_dates['event_name'] = None
        
        
        holidays = {
            'New Year': [(1, 1)],
            'Valentine': [(2, 14)],
            'Easter': [(4, 9), (4, 17)],  # Approximate
            'Memorial Day': [(5, 29), (5, 30), (5, 31)],
            'Independence Day': [(7, 4)],
            'Labor Day': [(9, 4), (9, 5), (9, 6)],
            'Halloween': [(10, 31)],
            'Thanksgiving': [(11, 23), (11, 24), (11, 25)],
            'Black Friday': [(11, 24), (11, 25)],
            'Cyber Monday': [(11, 27), (11, 28)],
            'Christmas': [(12, 24), (12, 25), (12, 26)]
        }
        
        for event_name, dates_list in holidays.items():
            for month, day in dates_list:
                mask = (df_dates['month'] == month) & (df_dates['day'] == day)
                df_dates.loc[mask, 'is_holiday'] = 1
                df_dates.loc[mask, 'event_name'] = event_name
        
        # Promotions (random weeks throughout the year)
        for year in df_dates['year'].unique():
            n_promotions = np.random.randint(8, 15)
            promotion_weeks = np.random.choice(range(1, 53), n_promotions, replace=False)
            
            for week in promotion_weeks:
                mask = (df_dates['year'] == year) & (df_dates['week_of_year'] == week)
                df_dates.loc[mask, 'is_promotion'] = 1
        
        
        mask = df_dates['month'].isin([8, 9])
        df_dates.loc[mask, 'is_back_to_school'] = 1
        
     
        mask = df_dates['month'].isin([11, 12])
        df_dates.loc[mask, 'is_holiday_season'] = 1
        
        return df_dates
    
    def generate_sales(self, df_products, df_dates):
        """Generate sales transactions"""
        print("Generating sales data...")
        
        sales_data = []
        
        for idx, product in df_products.iterrows():
            product_id = product['product_id']
            base_demand = product['base_demand']
            seasonality = product['seasonality_strength']
            trend = product['trend_coefficient']
            promo_sensitivity = product['promotion_sensitivity']
            dow_pattern = product['day_of_week_pattern']
            
            for date_idx, date_row in df_dates.iterrows():
                date = date_row['date']
                
                demand = base_demand
                
                
                days_since_start = (date - self.start_date).days
                demand = demand * (1 + trend * days_since_start)
                
               
                if seasonality > 0.3:
                   
                    month_factor = 1 + seasonality * np.sin(2 * np.pi * date_row['month'] / 12)
                    demand *= month_factor
                    
                   
                    week_factor = 1 + (seasonality * 0.3) * np.sin(2 * np.pi * date_row['week_of_year'] / 52)
                    demand *= week_factor
                
              
                if dow_pattern == 'weekend_high' and date_row['is_weekend']:
                    demand *= 1.4
                elif dow_pattern == 'weekday_high' and not date_row['is_weekend']:
                    demand *= 1.3
                
               
                if date_row['is_holiday']:
                    if product['category'] in ['Electronics', 'Toys', 'Clothing']:
                        demand *= np.random.uniform(2.0, 3.5)
                    else:
                        demand *= np.random.uniform(1.2, 1.8)
                
               
                if date_row['is_promotion']:
                    if np.random.random() < 0.6:  
                        demand *= promo_sensitivity
                
               
                demand *= np.random.uniform(0.8, 1.2)
                
                
                quantity_sold = max(0, int(round(demand)))
                
                
                if np.random.random() < 0.05:
                    quantity_sold = 0
                
                if quantity_sold > 0:
                    revenue = quantity_sold * product['price']
                    cost = quantity_sold * product['cost']
                    profit = revenue - cost
                    
                    sales_data.append({
                        'date': date,
                        'product_id': product_id,
                        'quantity_sold': quantity_sold,
                        'revenue': round(revenue, 2),
                        'cost': round(cost, 2),
                        'profit': round(profit, 2),
                        'discount_applied': 1 if date_row['is_promotion'] else 0,
                        'discount_percent': round(np.random.uniform(5, 30), 2) if date_row['is_promotion'] else 0
                    })
        
        df_sales = pd.DataFrame(sales_data)
        print(f"Generated {len(df_sales):,} sales records")
        
        return df_sales
    
    def generate_stock_levels(self, df_products, df_sales):
        """Generate stock level data"""
        print("Generating stock levels...")
        
        stock_data = []
        
        for product_id in df_products['product_id'].unique():
            product_info = df_products[df_products['product_id'] == product_id].iloc[0]
            product_sales = df_sales[df_sales['product_id'] == product_id].copy()
            product_sales = product_sales.sort_values('date')
            
           
            avg_daily_demand = product_sales['quantity_sold'].mean()
            current_stock = int(avg_daily_demand * np.random.randint(30, 90))
            
            for idx, sale in product_sales.iterrows():
                date = sale['date']
                qty_sold = sale['quantity_sold']
                
               
                stock_before = current_stock
                
                
                current_stock -= qty_sold
                
              
                reorder_point = int(avg_daily_demand * product_info['lead_time_days'] * 1.5)
                
                if current_stock < reorder_point:
                   
                    order_quantity = int(avg_daily_demand * 60)
                    current_stock += order_quantity
                    reorder = 1
                else:
                    order_quantity = 0
                    reorder = 0
                
                stock_data.append({
                    'date': date,
                    'product_id': product_id,
                    'stock_level': max(0, current_stock),
                    'stock_before_sale': stock_before,
                    'reorder_triggered': reorder,
                    'order_quantity': order_quantity
                })
        
        df_stock = pd.DataFrame(stock_data)
        print(f"Generated {len(df_stock):,} stock records")
        
        return df_stock
    
    def generate_all(self, save_path='data/synthetic/'):
        """Generate all datasets and save to CSV"""
        print(f"\n{'='*60}")
        print("SYNTHETIC RETAIL DATA GENERATION")
        print(f"{'='*60}\n")
        
        
        df_products = self.generate_products()
        df_dates = self.generate_date_features()
        df_sales = self.generate_sales(df_products, df_dates)
        df_stock = self.generate_stock_levels(df_products, df_sales)
        
        
        print(f"\nSaving data to {save_path}...")
        df_products.to_csv(f'{save_path}products.csv', index=False)
        df_dates.to_csv(f'{save_path}date_features.csv', index=False)
        df_sales.to_csv(f'{save_path}sales.csv', index=False)
        df_stock.to_csv(f'{save_path}stock_levels.csv', index=False)
        
        
        print(f"\n{'='*60}")
        print("DATA GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Products: {len(df_products):,}")
        print(f"Date Range: {df_dates['date'].min()} to {df_dates['date'].max()}")
        print(f"Total Days: {len(df_dates):,}")
        print(f"Sales Records: {len(df_sales):,}")
        print(f"Stock Records: {len(df_stock):,}")
        print(f"\nTotal Revenue: ${df_sales['revenue'].sum():,.2f}")
        print(f"Total Profit: ${df_sales['profit'].sum():,.2f}")
        print(f"Average Daily Sales: {df_sales.groupby('date')['quantity_sold'].sum().mean():,.0f} units")
        print(f"\nFiles saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return df_products, df_dates, df_sales, df_stock


def main():
    """Main function to generate data"""
    generator = SalesDataGenerator(
        n_products=100,
        n_years=3,
        start_date='2022-01-01',
        seed=42
    )
    
    df_products, df_dates, df_sales, df_stock = generator.generate_all(
        save_path='data/synthetic/'
    )
    
   
    print("\nPRODUCTS PREVIEW:")
    print(df_products.head())
    
    print("\n\nSALES PREVIEW:")
    print(df_sales.head(10))
    
    print("\n\nSTOCK LEVELS PREVIEW:")
    print(df_stock.head(10))


if __name__ == "__main__":
    main()
