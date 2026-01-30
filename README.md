# 📊 Sales and Stock Prediction System

## 🎯 Business Problem

Retail and e-commerce businesses face critical challenges:
- **Stockouts** lead to lost sales and customer dissatisfaction
- **Overstock** ties up capital and increases holding costs
- **Poor forecasting** results in inefficient inventory management
- **Lack of visibility** into future demand trends

This system solves these problems by providing:
- Accurate sales forecasting (7, 14, 30, 90 days ahead)
- Optimal stock level recommendations
- Automated reorder alerts
- What-if scenario analysis for business planning

## 💼 Business Impact

- **Reduce stockouts** by 60-80% through predictive alerts
- **Lower holding costs** by 20-35% via optimal stock levels
- **Improve forecast accuracy** to 85-95% MAPE for stable products
- **Increase service levels** to 95%+ while reducing inventory
- **Enable data-driven decisions** with scenario planning tools

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│  (Streamlit Dashboard / React Frontend)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Forecasting │  │  Inventory   │  │   Alert      │     │
│  │  Service     │  │  Optimizer   │  │   Engine     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Machine Learning Models                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ ARIMA/  │ │ Prophet │ │  LSTM   │ │ XGBoost │          │
│  │ SARIMA  │ │         │ │  /GRU   │ │         │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Data Layer (PostgreSQL/SQLite)                 │
│  - Historical Sales Data                                    │
│  - Product Information                                      │
│  - Stock Levels                                             │
│  - Model Predictions & Metadata                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

**Backend:**
- Python 3.9+
- FastAPI (REST API)
- SQLAlchemy (ORM)
- PostgreSQL/SQLite

**ML/Analytics:**
- pandas, numpy (data processing)
- scikit-learn (preprocessing, metrics)
- statsmodels (ARIMA/SARIMA)
- Prophet (Facebook's forecasting)
- TensorFlow/Keras (LSTM/GRU)
- XGBoost (gradient boosting)

**Frontend:**
- Streamlit (rapid prototyping)
- Plotly (interactive charts)
- Alternative: React + Recharts

**Deployment:**
- Docker (containerization)
- Heroku/Railway/AWS (hosting)
- Redis (optional caching)

## 📁 Project Structure

```
sales-stock-prediction/
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and engineered features
│   └── synthetic/              # Generated data
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_arima_sarima.ipynb
│   ├── 04_prophet.ipynb
│   ├── 05_lstm.ipynb
│   ├── 06_xgboost.ipynb
│   ├── 07_model_comparison.ipynb
│   └── 08_inventory_optimization.ipynb
│
├── src/
│   ├── data/
│   │   ├── data_generator.py   # Synthetic data generation
│   │   ├── preprocessing.py    # Data cleaning
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── base_model.py       # Abstract base class
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── lstm_model.py
│   │   ├── xgboost_model.py
│   │   └── ensemble.py
│   │
│   ├── inventory/
│   │   ├── eoq.py              # Economic Order Quantity
│   │   ├── reorder_point.py    # ROP calculation
│   │   ├── safety_stock.py
│   │   └── abc_analysis.py
│   │
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── routes/
│   │   │   ├── forecast.py
│   │   │   ├── inventory.py
│   │   │   └── analytics.py
│   │   └── schemas.py          # Pydantic models
│   │
│   ├── dashboard/
│   │   ├── app.py              # Streamlit main app
│   │   ├── pages/
│   │   │   ├── overview.py
│   │   │   ├── forecasting.py
│   │   │   ├── inventory.py
│   │   │   ├── analytics.py
│   │   │   └── whatif.py
│   │   └── components/         # Reusable UI components
│   │
│   └── utils/
│       ├── database.py
│       ├── metrics.py
│       ├── alerts.py
│       └── config.py
│
├── tests/
│   ├── test_models.py
│   ├── test_inventory.py
│   └── test_api.py
│
├── models/                     # Saved model artifacts
│   ├── arima/
│   ├── prophet/
│   ├── lstm/
│   └── xgboost/
│
├── config/
│   ├── config.yaml             # Application configuration
│   └── model_config.yaml       # Model hyperparameters
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   ├── business_case.md
│   └── deployment_guide.md
│
├── requirements.txt
├── setup.py
└── README.md
```


## 📊 Key Features

### 1. Sales Forecasting
- Multi-horizon predictions (7, 14, 30, 90 days)
- Multiple model comparison (ARIMA, Prophet, LSTM, XGBoost)
- Confidence intervals and uncertainty quantification
- Seasonal pattern detection
- Trend analysis

### 2. Inventory Optimization
- Economic Order Quantity (EOQ) calculation
- Reorder Point (ROP) with safety stock
- ABC analysis for product categorization
- Optimal stock level recommendations
- Dead stock identification

### 3. Interactive Dashboard
- Real-time sales forecast visualization
- Stock level tracking with alerts
- Product performance heatmaps
- Seasonal calendar view
- KPI monitoring (accuracy, efficiency, savings)

### 4. Alert System
- Low stock warnings (approaching reorder point)
- Overstock alerts (slow-moving inventory)
- Forecast accuracy degradation detection
- Seasonal spike predictions

### 5. What-If Analysis
- Promotional campaign impact simulation
- Lead time adjustment scenarios
- Demand increase/decrease planning
- Cost-benefit analysis

## 📈 Model Performance

Expected performance metrics for different models:

| Model | MAPE | RMSE | Training Time | Inference Speed |
|-------|------|------|---------------|-----------------|
| ARIMA/SARIMA | 15-25% | Medium | Fast | Very Fast |
| Prophet | 12-20% | Low | Fast | Fast |
| LSTM/GRU | 10-18% | Low | Slow | Medium |
| XGBoost | 12-22% | Medium | Medium | Very Fast |
| Ensemble | 10-15% | Lowest | Slow | Medium |

## 🎯 Business Metrics

### Inventory KPIs
- **Service Level**: Target 95%+ (orders fulfilled without stockout)
- **Stock Turnover Ratio**: 6-12x annually (industry dependent)
- **Days of Inventory**: 30-60 days optimal
- **Stockout Rate**: < 5%
- **Overstock Percentage**: < 15%

### Cost Savings
- Holding cost reduction: 20-35%
- Stockout cost reduction: 60-80%
- Working capital optimization: 15-25%
- Total inventory cost reduction: 25-40%

## 📚 API Endpoints

### Forecasting
```
POST   /api/v1/train                    # Train/retrain models
GET    /api/v1/forecast/{product_id}    # Get sales forecast
POST   /api/v1/forecast/batch           # Batch forecasting
GET    /api/v1/models/performance       # Model metrics
```

### Inventory
```
GET    /api/v1/stock/{product_id}              # Current stock status
GET    /api/v1/stock/recommendation/{id}       # Optimal stock level
GET    /api/v1/reorder/alerts                  # Reorder recommendations
GET    /api/v1/inventory/abc-analysis          # ABC categorization
GET    /api/v1/inventory/deadstock             # Dead stock report
```

### Analytics
```
GET    /api/v1/analytics/trends                # Trend analysis
GET    /api/v1/analytics/seasonality           # Seasonal patterns
GET    /api/v1/analytics/anomalies             # Anomaly detection
POST   /api/v1/analytics/whatif                # Scenario analysis
GET    /api/v1/analytics/kpis                  # Dashboard KPIs
```

Full API documentation available at `/docs` (Swagger UI)

## 🔬 Technical Deep Dive

### Time Series Models

**ARIMA/SARIMA**
- Best for: Stationary or trend-stationary data with clear patterns
- Pros: Fast, interpretable, works well with limited data
- Cons: Requires stationarity, struggles with multiple seasonalities

**Prophet**
- Best for: Daily data with strong seasonal effects and holidays
- Pros: Handles missing data, robust to outliers, easy to use
- Cons: May overfit on limited data, less flexible than neural networks

**LSTM/GRU**
- Best for: Complex patterns, long-term dependencies
- Pros: Captures non-linear relationships, learns from raw features
- Cons: Requires lots of data, computationally expensive, black box

**XGBoost**
- Best for: Tabular data with engineered features
- Pros: Fast, accurate, handles missing values, feature importance
- Cons: Requires good feature engineering, less suited for raw sequences

### Inventory Formulas

**Economic Order Quantity (EOQ)**
```
EOQ = √(2 × D × S / H)

Where:
D = Annual demand
S = Order cost per order
H = Holding cost per unit per year
```

**Reorder Point (ROP)**
```
ROP = (Average Daily Demand × Lead Time) + Safety Stock

Safety Stock = Z × σ × √LT

Where:
Z = Service level factor (e.g., 1.65 for 95%)
σ = Standard deviation of daily demand
LT = Lead time in days
```

**ABC Analysis**
- A items: Top 20% products → 80% revenue (tight control)
- B items: Next 30% products → 15% revenue (moderate control)
- C items: Bottom 50% products → 5% revenue (loose control)



## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 👨‍💻 Author

Built by Siddhant Sahare as a portfolio project demonstrating:
- End-to-end ML system design
- Time series forecasting expertise
- Full-stack development
- Business problem solving
- Production deployment

Note: This is a portfolio project demonstrating technical skills. For production use, ensure proper data security, model monitoring, and compliance with business requirements.
