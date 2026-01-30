# Sales and Stock Prediction System - Complete Implementation Guide

## 📚 Table of Contents
1. [Project Setup](#phase-0-project-setup)
2. [Phase 1: Data & Exploration](#phase-1-data--exploration)
3. [Phase 2: Forecasting Models](#phase-2-forecasting-models)
4. [Phase 3: Inventory Optimization](#phase-3-inventory-optimization)
5. [Phase 4: Application Development](#phase-4-application-development)
6. [Phase 5: Deployment](#phase-5-deployment)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Phase 0: Project Setup

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installations
python -c "import pandas, numpy, sklearn, prophet, tensorflow, xgboost; print('✓ All imports successful')"
```

### Directory Structure
The project follows this organization:
- `data/`: Raw, processed, and synthetic datasets
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `src/`: Production code (data, models, API, dashboard)
- `models/`: Saved model artifacts
- `tests/`: Unit and integration tests
- `docs/`: Documentation
- `config/`: Configuration files

---

## Phase 1: Data & Exploration

### Step 1.1: Generate Synthetic Data ✓ COMPLETE

We've successfully generated realistic retail data with:
- **100 products** across 10 categories
- **3 years** of daily sales (2022-2025)
- **104,043 sales records** with revenue of $5.8B
- **Seasonal patterns** (weekly, monthly, yearly)
- **Promotional effects** and holidays
- **Stock level tracking** with reorder logic

**Files Created:**
- `src/data/data_generator.py` - Synthetic data generator
- `data/synthetic/products.csv` - Product master data
- `data/synthetic/sales.csv` - Historical sales transactions
- `data/synthetic/stock_levels.csv` - Inventory levels
- `data/synthetic/date_features.csv` - Calendar features

**Key Statistics:**
- Average daily sales: 20,373 units
- Total revenue: $5.82 billion
- Total profit: $2.65 billion
- Date range: Jan 1, 2022 - Dec 31, 2024

### Step 1.2: Exploratory Data Analysis (EDA) ✓ COMPLETE

Created comprehensive EDA notebook with:
- Data quality checks (no missing values)
- Distribution analysis (products, prices, margins)
- Time series visualization
- Seasonality detection (monthly, weekly patterns)
- Category performance analysis
- Top products identification
- Stock level analysis
- Correlation analysis

**Notebook:** `notebooks/01_eda.ipynb`

**Key Findings:**
1. Strong seasonality in sales (holiday peaks in Nov-Dec)
2. Weekend vs weekday patterns vary by product category
3. Electronics and Toys are top revenue categories
4. Clear promotional impact visible in data
5. Lead times range from 3-30 days across suppliers

### Step 1.3: Feature Engineering

Next, we'll create features for time series forecasting:

**Time-based features:**
- Lag features (t-1, t-7, t-14, t-30, t-90)
- Rolling statistics (mean, std, min, max)
- Exponential moving averages
- Seasonal indicators (month, quarter, day_of_week)

**Calendar features:**
- Is_holiday, is_weekend, is_month_end
- Days_to_next_holiday
- Week_of_year, month_of_year

**Product-specific features:**
- Price, category, supplier
- Historical average sales
- Volatility metrics
- Promotion history

**Implementation:**
```python
# notebooks/02_feature_engineering.ipynb
# src/data/feature_engineering.py
```

---

## Phase 2: Forecasting Models

### Model Selection Strategy

We'll implement 4 different model types and compare:

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **ARIMA/SARIMA** | Stable, seasonal products | Fast, interpretable | Requires stationarity |
| **Prophet** | Products with strong seasonality | Handles missing data, robust | May overfit limited data |
| **LSTM/GRU** | Complex patterns | Learns non-linear patterns | Needs lots of data, slow |
| **XGBoost** | Tabular features | Fast, accurate, explainable | Requires feature engineering |

### Step 2.1: Baseline Models

Before complex models, establish baselines:
1. **Naive Forecast**: Tomorrow = Today
2. **Moving Average**: Average of last N days
3. **Seasonal Naive**: Same day last week/month/year

These provide minimum performance benchmarks.

### Step 2.2: ARIMA/SARIMA Implementation

**Notebook:** `notebooks/03_arima_sarima.ipynb`
**Module:** `src/models/arima_model.py`

**Process:**
1. Check stationarity (ADF test)
2. Apply differencing if needed
3. ACF/PACF plots for parameter selection
4. Grid search for optimal (p,d,q)(P,D,Q,s) parameters
5. Fit model per product
6. Generate forecasts with confidence intervals

**Code Example:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
model = SARIMAX(
    train_data,
    order=(1, 1, 1),  # (p, d, q)
    seasonal_order=(1, 1, 1, 7),  # (P, D, Q, s) - weekly seasonality
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
forecast = results.forecast(steps=30)
```

### Step 2.3: Prophet Implementation

**Notebook:** `notebooks/04_prophet.ipynb`
**Module:** `src/models/prophet_model.py`

**Features:**
- Automatic seasonality detection
- Holiday effects
- Multiple seasonalities (weekly, yearly)
- Custom regressors (promotions, price changes)

**Code Example:**
```python
from prophet import Prophet

# Prepare data
df_prophet = df[['date', 'quantity_sold']].rename(
    columns={'date': 'ds', 'quantity_sold': 'y'}
)

# Create and fit model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

# Add holidays
model.add_country_holidays(country_name='US')

# Add custom regressors
model.add_regressor('is_promotion')

model.fit(df_prophet)

# Forecast
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

### Step 2.4: LSTM/GRU Implementation

**Notebook:** `notebooks/05_lstm.ipynb`
**Module:** `src/models/lstm_model.py`

**Architecture:**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(sequence_length, n_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output: next day sales
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Prepare sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Train
model = build_lstm_model(sequence_length=30, n_features=5)
history = model.fit(X_train, y_train, 
                   epochs=100, 
                   batch_size=32,
                   validation_split=0.2,
                   callbacks=[early_stopping])
```

### Step 2.5: XGBoost Implementation

**Notebook:** `notebooks/06_xgboost.ipynb`
**Module:** `src/models/xgboost_model.py`

**Features Used:**
- Lag features (1, 7, 14, 30 days)
- Rolling statistics
- Calendar features
- Product features
- External factors (promotions, holidays)

**Code Example:**
```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# Prepare features
features = [
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_std_7',
    'day_of_week', 'month', 'is_weekend',
    'is_holiday', 'is_promotion',
    'price', 'category_encoded'
]

X_train = df_train[features]
y_train = df_train['quantity_sold']

# Train model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=15)
plt.show()
```

### Step 2.6: Model Comparison

**Notebook:** `notebooks/07_model_comparison.ipynb`

**Evaluation Metrics:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }
```

**Comparison Framework:**
- Train all models on same data split
- Evaluate on hold-out test set
- Compare by product category
- Analyze errors by time period
- Consider computational cost

**Expected Results:**

| Model | MAPE | RMSE | Training Time | Inference Speed |
|-------|------|------|---------------|-----------------|
| Naive | 30-40% | High | Instant | Instant |
| ARIMA | 15-25% | Medium | Fast | Very Fast |
| Prophet | 12-20% | Low | Fast | Fast |
| LSTM | 10-18% | Low | Slow | Medium |
| XGBoost | 12-22% | Medium | Medium | Very Fast |

### Step 2.7: Ensemble Model (Optional)

Combine predictions from multiple models:

```python
def ensemble_forecast(arima_pred, prophet_pred, lstm_pred, xgb_pred, weights=None):
    if weights is None:
        weights = [0.2, 0.3, 0.3, 0.2]  # Based on validation performance
    
    ensemble = (
        weights[0] * arima_pred +
        weights[1] * prophet_pred +
        weights[2] * lstm_pred +
        weights[3] * xgb_pred
    )
    
    return ensemble
```

---

## Phase 3: Inventory Optimization

### Step 3.1: Economic Order Quantity (EOQ)

**Module:** `src/inventory/eoq.py`

**Formula:**
```
EOQ = √(2 × D × S / H)

Where:
D = Annual demand (units)
S = Ordering cost per order ($)
H = Holding cost per unit per year ($)
```

**Implementation:**
```python
import numpy as np

class EOQCalculator:
    def __init__(self, annual_demand, order_cost, holding_cost_percent, unit_cost):
        self.annual_demand = annual_demand
        self.order_cost = order_cost
        self.holding_cost = unit_cost * holding_cost_percent
    
    def calculate_eoq(self):
        eoq = np.sqrt(
            (2 * self.annual_demand * self.order_cost) / self.holding_cost
        )
        return round(eoq, 0)
    
    def calculate_total_cost(self, order_quantity):
        # Number of orders per year
        n_orders = self.annual_demand / order_quantity
        
        # Ordering cost
        ordering_cost = n_orders * self.order_cost
        
        # Holding cost
        holding_cost = (order_quantity / 2) * self.holding_cost
        
        # Total cost
        total_cost = ordering_cost + holding_cost
        
        return total_cost
    
    def get_optimal_order_policy(self):
        eoq = self.calculate_eoq()
        n_orders = self.annual_demand / eoq
        time_between_orders = 365 / n_orders
        total_cost = self.calculate_total_cost(eoq)
        
        return {
            'eoq': eoq,
            'orders_per_year': round(n_orders, 2),
            'days_between_orders': round(time_between_orders, 1),
            'annual_total_cost': round(total_cost, 2)
        }
```

### Step 3.2: Reorder Point (ROP) Calculation

**Module:** `src/inventory/reorder_point.py`

**Formula:**
```
ROP = (Average Daily Demand × Lead Time) + Safety Stock

Safety Stock = Z × σ × √LT

Where:
Z = Service level factor (e.g., 1.65 for 95%, 1.96 for 97.5%, 2.33 for 99%)
σ = Standard deviation of daily demand
LT = Lead time in days
```

**Implementation:**
```python
from scipy import stats
import numpy as np

class ReorderPointCalculator:
    def __init__(self, daily_demand_mean, daily_demand_std, lead_time_days, service_level=0.95):
        self.demand_mean = daily_demand_mean
        self.demand_std = daily_demand_std
        self.lead_time = lead_time_days
        self.service_level = service_level
        
        # Z-score for service level
        self.z_score = stats.norm.ppf(service_level)
    
    def calculate_safety_stock(self):
        safety_stock = self.z_score * self.demand_std * np.sqrt(self.lead_time)
        return round(safety_stock, 0)
    
    def calculate_reorder_point(self):
        expected_demand = self.demand_mean * self.lead_time
        safety_stock = self.calculate_safety_stock()
        rop = expected_demand + safety_stock
        return round(rop, 0)
    
    def get_inventory_policy(self):
        rop = self.calculate_reorder_point()
        safety_stock = self.calculate_safety_stock()
        
        return {
            'reorder_point': rop,
            'safety_stock': safety_stock,
            'expected_demand_during_lead_time': round(self.demand_mean * self.lead_time, 0),
            'service_level': f"{self.service_level * 100}%",
            'stockout_probability': f"{(1 - self.service_level) * 100}%"
        }
```

### Step 3.3: ABC Analysis

**Module:** `src/inventory/abc_analysis.py`

**Classification:**
- **A items**: Top 20% of products → 80% of revenue (tight control)
- **B items**: Next 30% of products → 15% of revenue (moderate control)
- **C items**: Bottom 50% of products → 5% of revenue (loose control)

**Implementation:**
```python
import pandas as pd

class ABCAnalyzer:
    def __init__(self, df_sales):
        self.df = df_sales
    
    def classify_products(self):
        # Calculate total revenue per product
        product_revenue = self.df.groupby('product_id')['revenue'].sum().reset_index()
        product_revenue = product_revenue.sort_values('revenue', ascending=False)
        
        # Calculate cumulative percentage
        total_revenue = product_revenue['revenue'].sum()
        product_revenue['revenue_percent'] = (product_revenue['revenue'] / total_revenue) * 100
        product_revenue['cumulative_percent'] = product_revenue['revenue_percent'].cumsum()
        
        # Classify
        def classify(cum_percent):
            if cum_percent <= 80:
                return 'A'
            elif cum_percent <= 95:
                return 'B'
            else:
                return 'C'
        
        product_revenue['abc_class'] = product_revenue['cumulative_percent'].apply(classify)
        
        return product_revenue
    
    def get_summary(self):
        classified = self.classify_products()
        
        summary = classified.groupby('abc_class').agg({
            'product_id': 'count',
            'revenue': 'sum'
        }).reset_index()
        
        summary.columns = ['ABC_Class', 'Product_Count', 'Total_Revenue']
        summary['Revenue_Percent'] = (summary['Total_Revenue'] / summary['Total_Revenue'].sum()) * 100
        
        return summary
```

### Step 3.4: Safety Stock Optimization

Consider:
- Demand variability
- Lead time variability
- Service level targets
- Cost trade-offs

**Advanced formula with lead time variability:**
```python
def calculate_safety_stock_advanced(avg_demand, demand_std, avg_lead_time, lead_time_std, service_level):
    z = stats.norm.ppf(service_level)
    
    # Variance of demand during lead time
    variance = (avg_lead_time * demand_std**2) + (avg_demand**2 * lead_time_std**2)
    
    safety_stock = z * np.sqrt(variance)
    return safety_stock
```

### Step 3.5: Stock Optimization Integration

**Notebook:** `notebooks/08_inventory_optimization.ipynb`

Combine forecasting + inventory policies:

```python
def optimize_inventory(product_id, forecast_demand, product_info):
    # Get forecast statistics
    avg_daily_demand = forecast_demand.mean()
    std_daily_demand = forecast_demand.std()
    
    # Product-specific parameters
    lead_time = product_info['lead_time_days']
    unit_cost = product_info['cost']
    order_cost = 50  # Assumed
    holding_cost_percent = 0.25  # 25% of unit cost per year
    
    # Calculate EOQ
    annual_demand = avg_daily_demand * 365
    eoq_calc = EOQCalculator(annual_demand, order_cost, holding_cost_percent, unit_cost)
    eoq_policy = eoq_calc.get_optimal_order_policy()
    
    # Calculate ROP
    rop_calc = ReorderPointCalculator(avg_daily_demand, std_daily_demand, lead_time, service_level=0.95)
    rop_policy = rop_calc.get_inventory_policy()
    
    return {
        'product_id': product_id,
        'forecast_daily_demand': round(avg_daily_demand, 2),
        'demand_volatility': round(std_daily_demand, 2),
        **eoq_policy,
        **rop_policy
    }
```

---

## Phase 4: Application Development

### Step 4.1: FastAPI Backend

**Module:** `src/api/main.py`

**Structure:**
```
src/api/
├── main.py              # FastAPI app initialization
├── routes/
│   ├── forecast.py      # Forecasting endpoints
│   ├── inventory.py     # Inventory endpoints
│   └── analytics.py     # Analytics endpoints
├── schemas.py           # Pydantic models
└── dependencies.py      # Shared dependencies
```

**Main Application:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import forecast, inventory, analytics

app = FastAPI(
    title="Sales & Stock Prediction API",
    description="ML-powered sales forecasting and inventory optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecast.router, prefix="/api/v1/forecast", tags=["Forecasting"])
app.include_router(inventory.router, prefix="/api/v1/inventory", tags=["Inventory"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])

@app.get("/")
def read_root():
    return {
        "message": "Sales & Stock Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

**Forecast Endpoint Example:**
```python
# src/api/routes/forecast.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

router = APIRouter()

class ForecastRequest(BaseModel):
    product_id: str
    horizon: int = 30
    model: Optional[str] = "prophet"

class ForecastResponse(BaseModel):
    product_id: str
    forecast_dates: List[str]
    forecast_values: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    model_used: str
    accuracy_metrics: dict

@router.post("/predict", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    try:
        # Load model
        model = load_model(request.product_id, request.model)
        
        # Generate forecast
        forecast = model.predict(horizon=request.horizon)
        
        return ForecastResponse(
            product_id=request.product_id,
            forecast_dates=forecast['dates'].tolist(),
            forecast_values=forecast['predictions'].tolist(),
            lower_bound=forecast['lower'].tolist(),
            upper_bound=forecast['upper'].tolist(),
            model_used=request.model,
            accuracy_metrics=forecast['metrics']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 4.2: Streamlit Dashboard

**Module:** `src/dashboard/app.py`

**Multi-page Structure:**
```python
import streamlit as st
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(
    page_title="Sales & Stock Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Overview", "Forecasting", "Inventory", "Analytics", "What-If"],
        icons=["house", "graph-up", "box-seam", "bar-chart", "sliders"],
        menu_icon="cast",
        default_index=0,
    )

# Page routing
if selected == "Overview":
    from pages import overview
    overview.show()
elif selected == "Forecasting":
    from pages import forecasting
    forecasting.show()
elif selected == "Inventory":
    from pages import inventory
    inventory.show()
elif selected == "Analytics":
    from pages import analytics
    analytics.show()
elif selected == "What-If":
    from pages import whatif
    whatif.show()
```

**Overview Page Example:**
```python
# src/dashboard/pages/overview.py
import streamlit as st
import plotly.express as px
import pandas as pd

def show():
    st.title("📊 Sales & Stock Prediction - Overview")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Revenue (YTD)",
            value="$5.82M",
            delta="+12.3%"
        )
    
    with col2:
        st.metric(
            label="Stock-out Rate",
            value="3.2%",
            delta="-1.5%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Forecast Accuracy (MAPE)",
            value="14.5%",
            delta="-2.1%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Inventory Turnover",
            value="8.2x",
            delta="+0.4x"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Sales Trend")
        # Plot interactive chart
        fig = px.line(df_daily_sales, x='date', y='revenue')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Products by Revenue")
        fig = px.bar(df_top_products, x='product_id', y='revenue')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Alerts
    st.subheader("🚨 Recent Alerts")
    alerts = get_recent_alerts()
    st.dataframe(alerts, use_container_width=True)
```

### Step 4.3: Alert System

**Module:** `src/utils/alerts.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import List
import pandas as pd

class AlertType(Enum):
    LOW_STOCK = "Low Stock Warning"
    OVERSTOCK = "Overstock Alert"
    FORECAST_DEGRADATION = "Forecast Accuracy Drop"
    SEASONAL_SPIKE = "Seasonal Demand Spike"

@dataclass
class Alert:
    product_id: str
    alert_type: AlertType
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_value: float
    threshold: float
    recommendation: str
    created_at: str

class AlertEngine:
    def __init__(self, df_sales, df_stock, df_forecast):
        self.df_sales = df_sales
        self.df_stock = df_stock
        self.df_forecast = df_forecast
        self.alerts = []
    
    def check_low_stock(self, reorder_points):
        for product_id, rop in reorder_points.items():
            current_stock = self.df_stock[
                self.df_stock['product_id'] == product_id
            ]['stock_level'].iloc[-1]
            
            if current_stock < rop:
                severity = 'critical' if current_stock < rop * 0.5 else 'high'
                
                self.alerts.append(Alert(
                    product_id=product_id,
                    alert_type=AlertType.LOW_STOCK,
                    severity=severity,
                    message=f"Stock level ({current_stock}) below reorder point ({rop})",
                    metric_value=current_stock,
                    threshold=rop,
                    recommendation=f"Order immediately. Recommended quantity: {self.calculate_order_qty(product_id)}",
                    created_at=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
    
    def check_forecast_accuracy(self, accuracy_threshold=0.20):
        # Check if forecast accuracy has degraded
        for product_id in self.df_forecast['product_id'].unique():
            recent_mape = self.calculate_recent_mape(product_id)
            
            if recent_mape > accuracy_threshold:
                self.alerts.append(Alert(
                    product_id=product_id,
                    alert_type=AlertType.FORECAST_DEGRADATION,
                    severity='medium',
                    message=f"Forecast accuracy degraded (MAPE: {recent_mape:.1%})",
                    metric_value=recent_mape,
                    threshold=accuracy_threshold,
                    recommendation="Retrain model with recent data",
                    created_at=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
    
    def get_all_alerts(self) -> List[Alert]:
        return sorted(self.alerts, key=lambda x: x.severity, reverse=True)
```

### Step 4.4: What-If Simulator

**Module:** `src/dashboard/pages/whatif.py`

```python
import streamlit as st
import plotly.graph_objects as go

def show():
    st.title("🎯 What-If Analysis Simulator")
    
    st.markdown("""
    Simulate different scenarios to understand impact on inventory and sales.
    Adjust parameters below to see predicted outcomes.
    """)
    
    # Scenario controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scenario Parameters")
        
        demand_change = st.slider(
            "Expected Demand Change (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=5
        )
        
        lead_time_change = st.slider(
            "Lead Time Change (days)",
            min_value=-10,
            max_value=20,
            value=0,
            step=1
        )
        
        service_level = st.slider(
            "Target Service Level (%)",
            min_value=85,
            max_value=99,
            value=95,
            step=1
        )
        
        promo_intensity = st.slider(
            "Promotional Intensity",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1
        )
    
    with col2:
        st.subheader("Predicted Impact")
        
        # Run simulation
        results = run_whatif_simulation(
            demand_change=demand_change,
            lead_time_change=lead_time_change,
            service_level=service_level / 100,
            promo_intensity=promo_intensity
        )
        
        # Display results
        st.metric("New Reorder Point", f"{results['new_rop']:.0f} units")
        st.metric("New Safety Stock", f"{results['new_safety_stock']:.0f} units")
        st.metric("Expected Stock-out Rate", f"{results['stockout_rate']:.1%}")
        st.metric("Holding Cost Change", f"${results['cost_change']:,.2f}", 
                 delta=f"{results['cost_change_pct']:.1%}")
    
    # Visualization
    st.subheader("Impact Visualization")
    
    fig = go.Figure()
    
    # Current vs New scenario
    categories = ['Reorder Point', 'Safety Stock', 'Average Stock', 'Total Cost']
    current_values = [results['current_rop'], results['current_safety'], 
                     results['current_avg_stock'], results['current_cost']]
    new_values = [results['new_rop'], results['new_safety'],
                 results['new_avg_stock'], results['new_cost']]
    
    fig.add_trace(go.Bar(name='Current', x=categories, y=current_values))
    fig.add_trace(go.Bar(name='New Scenario', x=categories, y=new_values))
    
    fig.update_layout(barmode='group', title='Current vs New Scenario Comparison')
    st.plotly_chart(fig, use_container_width=True)
```

---

## Phase 5: Deployment

### Step 5.1: Dockerization

**File:** `docker/Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 8501

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File:** `docker/docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/sales_prediction
    depends_on:
      - db
    volumes:
      - ../models:/app/models
      - ../data:/app/data
  
  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    command: streamlit run src/dashboard/app.py --server.port=8501
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
  
  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=sales_prediction
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Step 5.2: Model Retraining Pipeline

**Module:** `src/utils/model_trainer.py`

```python
import schedule
import time
from datetime import datetime

class ModelRetrainingPipeline:
    def __init__(self, retrain_frequency='weekly'):
        self.frequency = retrain_frequency
        
    def retrain_all_models(self):
        print(f"[{datetime.now()}] Starting model retraining...")
        
        # 1. Fetch latest data
        df_sales = self.fetch_latest_sales_data()
        
        # 2. Retrain each model type
        models_to_train = ['arima', 'prophet', 'lstm', 'xgboost']
        
        for model_type in models_to_train:
            print(f"Training {model_type}...")
            new_model = self.train_model(df_sales, model_type)
            
            # 3. Evaluate on validation set
            performance = self.evaluate_model(new_model)
            
            # 4. Compare with current production model
            if self.is_better_than_current(performance, model_type):
                print(f"New {model_type} model is better! Deploying...")
                self.deploy_model(new_model, model_type)
            else:
                print(f"Current {model_type} model still performs better. Keeping it.")
        
        print(f"[{datetime.now()}] Retraining completed!")
    
    def schedule_retraining(self):
        if self.frequency == 'daily':
            schedule.every().day.at("02:00").do(self.retrain_all_models)
        elif self.frequency == 'weekly':
            schedule.every().sunday.at("02:00").do(self.retrain_all_models)
        elif self.frequency == 'monthly':
            schedule.every(30).days.do(self.retrain_all_models)
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    pipeline = ModelRetrainingPipeline(retrain_frequency='weekly')
    pipeline.schedule_retraining()
```

### Step 5.3: Cloud Deployment (Heroku Example)

**File:** `Procfile`
```
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
worker: streamlit run src/dashboard/app.py --server.port $PORT
```

**File:** `runtime.txt`
```
python-3.9.16
```

**Deployment Commands:**
```bash
# Install Heroku CLI
# Login
heroku login

# Create app
heroku create your-sales-prediction-app

# Add PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# Set environment variables
heroku config:set DATABASE_URL=<your-db-url>

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1 worker=1
```

---

## Best Practices

### Code Organization
1. **Modular design**: Each component (data, models, API) is independent
2. **Configuration management**: Use YAML/ENV files for settings
3. **Version control**: Git with meaningful commit messages
4. **Documentation**: Docstrings for all functions and classes

### Model Management
1. **Versioning**: Track model versions with metadata
2. **A/B testing**: Compare new models against production before deployment
3. **Monitoring**: Track prediction accuracy over time
4. **Fallback**: Have baseline models as backup

### Performance Optimization
1. **Caching**: Use Redis for frequently requested forecasts
2. **Batch processing**: Process multiple products together
3. **Lazy loading**: Load models only when needed
4. **Database indexing**: Index on product_id and date columns

### Security
1. **API authentication**: Use JWT tokens
2. **Rate limiting**: Prevent abuse
3. **Input validation**: Sanitize all inputs
4. **HTTPS**: Always use secure connections

---

## Troubleshooting

### Common Issues

**Issue: ARIMA fails to converge**
- Solution: Try different (p,d,q) parameters or use auto_arima from pmdarima

**Issue: LSTM overfitting**
- Solution: Add dropout layers, reduce model complexity, use early stopping

**Issue: Slow API responses**
- Solution: Implement caching, use async processing, optimize database queries

**Issue: Forecast accuracy degrading**
- Solution: Retrain with recent data, check for data drift, adjust features

---

## Next Steps

After completing implementation:

1. **Testing**: Write comprehensive unit and integration tests
2. **Documentation**: Complete API docs and user guides
3. **Presentation**: Create slides for portfolio showcase
4. **Blog Post**: Write technical article explaining approach
5. **Demo Video**: Record walkthrough of dashboard
6. **GitHub README**: Polish with badges, screenshots, setup guide

---

## Resources

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

*This guide provides a complete roadmap. Implement phase by phase, test thoroughly, and iterate based on results.*
