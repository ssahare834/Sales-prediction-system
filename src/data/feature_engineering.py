import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngineer:
    """
    Engineer features for time series forecasting
    """
    
    def __init__(self, df_sales, df_products, df_dates):
        """
        Initialize with sales, product, and date dataframes
        
        Args:
            df_sales: DataFrame with sales transactions
            df_products: DataFrame with product information
            df_dates: DataFrame with date features
        """
        self.df_sales = df_sales.copy()
        self.df_products = df_products.copy()
        self.df_dates = df_dates.copy()
        
      
        self.df_sales['date'] = pd.to_datetime(self.df_sales['date'])
        self.df_dates['date'] = pd.to_datetime(self.df_dates['date'])
    
    def create_daily_aggregates(self, product_id=None):
        """
        Aggregate sales to daily level
        
        Args:
            product_id: Specific product ID or None for all products
        
        Returns:
            DataFrame with daily sales aggregates
        """
        if product_id:
            df = self.df_sales[self.df_sales['product_id'] == product_id].copy()
        else:
            df = self.df_sales.copy()
        
        
        daily = df.groupby(['date', 'product_id']).agg({
            'quantity_sold': 'sum',
            'revenue': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        all_dates = pd.date_range(
            start=daily['date'].min(),
            end=daily['date'].max(),
            freq='D'
        )
        
        if product_id:
            date_df = pd.DataFrame({'date': all_dates, 'product_id': product_id})
        else:
            products = daily['product_id'].unique()
            date_df = pd.DataFrame([
                {'date': date, 'product_id': prod}
                for date in all_dates
                for prod in products
            ])
        
        daily = date_df.merge(daily, on=['date', 'product_id'], how='left')
        daily = daily.fillna(0)
        daily = daily.sort_values(['product_id', 'date'])
        
        return daily
    
    def add_lag_features(self, df, target_col='quantity_sold', lags=[1, 7, 14, 30, 90]):
        """
        Add lag features (previous time steps)
        
        Args:
            df: DataFrame with date and target column
            target_col: Column to create lags for
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('product_id')[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df, target_col='quantity_sold', windows=[7, 14, 30]):
        """
        Add rolling window statistics
        
        Args:
            df: DataFrame with date and target column
            target_col: Column to calculate rolling stats for
            windows: List of window sizes
        
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby('product_id')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            df[f'rolling_std_{window}'] = df.groupby('product_id')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            df[f'rolling_min_{window}'] = df.groupby('product_id')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            
            df[f'rolling_max_{window}'] = df.groupby('product_id')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        
        return df
    
    def add_ewm_features(self, df, target_col='quantity_sold', spans=[7, 30]):
        """
        Add exponential weighted moving averages
        
        Args:
            df: DataFrame with date and target column
            target_col: Column to calculate EWM for
            spans: List of span periods
        
        Returns:
            DataFrame with EWM features
        """
        df = df.copy()
        
        for span in spans:
            df[f'ewm_{span}'] = df.groupby('product_id')[target_col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
        
        return df
    
    def add_calendar_features(self, df):
        """
        Add calendar-based features
        
        Args:
            df: DataFrame with date column
        
        Returns:
            DataFrame with calendar features
        """
        df = df.copy()
        
        df = df.merge(self.df_dates, on='date', how='left')
        
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def add_product_features(self, df):
        """
        Add product-specific features
        
        Args:
            df: DataFrame with product_id
        
        Returns:
            DataFrame with product features
        """
        df = df.copy()
        
        product_cols = ['product_id', 'category', 'price', 'cost', 
                       'margin_percent', 'lead_time_days']
        df = df.merge(self.df_products[product_cols], on='product_id', how='left')
        
        df['category_encoded'] = pd.Categorical(df['category']).codes
        
        return df
    
    def add_trend_features(self, df):
        """
        Add trend features
        
        Args:
            df: DataFrame with date column
        
        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        df['week_number'] = df['date'].dt.isocalendar().week
        
        return df
    
    def create_all_features(self, product_id=None):
        """
        Create all features for modeling
        
        Args:
            product_id: Specific product or None for all products
        
        Returns:
            DataFrame with all features
        """
        print("Creating daily aggregates...")
        df = self.create_daily_aggregates(product_id)
        
        print("Adding lag features...")
        df = self.add_lag_features(df, target_col='quantity_sold', lags=[1, 7, 14, 30])
        
        print("Adding rolling features...")
        df = self.add_rolling_features(df, target_col='quantity_sold', windows=[7, 14, 30])
        
        print("Adding EWM features...")
        df = self.add_ewm_features(df, target_col='quantity_sold', spans=[7, 30])
        
        print("Adding calendar features...")
        df = self.add_calendar_features(df)
        
        print("Adding product features...")
        df = self.add_product_features(df)
        
        print("Adding trend features...")
        df = self.add_trend_features(df)
        
        df = df.dropna()
        
        print(f"Feature engineering complete! Shape: {df.shape}")
        
        return df
    
    def get_feature_names(self):
        """
        Get list of all feature column names
        
        Returns:
            List of feature names
        """
        features = [
            'lag_1', 'lag_7', 'lag_14', 'lag_30',
            
            'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
            'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
            'rolling_mean_30', 'rolling_std_30', 'rolling_min_30', 'rolling_max_30',
            
            'ewm_7', 'ewm_30',
            
            'day_of_week', 'month', 'day', 'quarter',
            'is_weekend', 'is_month_start', 'is_month_end',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'is_holiday', 'is_promotion',
            
            'price', 'cost', 'margin_percent', 'lead_time_days',
            'category_encoded',
            
            'days_since_start', 'week_number'
        ]
        
        return features


def main():
    """Test feature engineering"""
    print("Loading data...")
    df_products = pd.read_csv('data/synthetic/products.csv')
    df_sales = pd.read_csv('data/synthetic/sales.csv')
    df_dates = pd.read_csv('data/synthetic/date_features.csv')
    
    print(f"Products: {len(df_products)}")
    print(f"Sales records: {len(df_sales)}")
    
    print("\n" + "="*60)
    print("Creating features for SKU_0001...")
    print("="*60)
    
    engineer = TimeSeriesFeatureEngineer(df_sales, df_products, df_dates)
    df_features = engineer.create_all_features(product_id='SKU_0001')
    
    print("\nFeature Summary:")
    print(df_features.head())
    print(f"\nShape: {df_features.shape}")
    print(f"\nColumns: {df_features.columns.tolist()}")
    
    df_features.to_csv('data/processed/features_SKU_0001.csv', index=False)
    print(f"\nSaved to: data/processed/features_SKU_0001.csv")


if __name__ == "__main__":
    main()
