# 📦 Last-Mile Delivery Forecasting & Route Optimization

## Overview
This project simulates an urban last-mile delivery network and applies predictive modeling and prescriptive optimization to improve delivery performance while minimizing operational cost.

The system forecasts hourly delivery demand by zone, models traffic-adjusted service times, and optimizes driver allocation to balance cost, capacity, and service-level agreements (SLAs).

---

## Key Business Questions
- How many deliveries will occur by zone and hour?
- How should drivers be allocated to minimize late deliveries?
- What is the cost vs SLA tradeoff under different demand scenarios?

---

## Data
- Synthetic delivery orders (60 days, hourly resolution)
- Real NYC geography via OpenStreetMap
- Traffic-adjusted service time simulation

---

## Methods

### Predictive Modeling
- Gradient Boosting Regression for hourly demand forecasting
- Time-based features (hour of day, day of week, weekend indicator)
- Zone-level demand segmentation

### Prescriptive Optimization
- Linear optimization using Google OR-Tools
- Decision variables: drivers per zone per hour
- Constraints: driver capacity and forecasted demand
- Objective: minimize labor cost and late-delivery penalties

### KPIs
- On-time delivery rate
- Total delivery cost
- Driver utilization
- Late order volume

---

## Results
- Optimized driver allocation reduced late deliveries while controlling labor cost
- Explicit cost–service tradeoff quantified for operational decision-making
- Outputs structured for executive dashboards and scenario analysis

---

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- OR-Tools
- OpenStreetMap (osmnx)
- Matplotlib, Seaborn

---

## Repository Structure
```last-mile-delivery-forecasting/
│
├── data/
│   ├── raw/
│   │   └── synthetic_orders.csv
│   ├── processed/
│   │   ├── hourly_demand.csv
│   │   ├── demand_forecasts.csv
│   │   ├── optimization_results.csv
│   │   └── kpis_summary.csv
│
├── notebooks/
│   └── 01_last_mile_forecasting_and_optimization.ipynb
│
├── src/
│   ├── data_generation.py
│   ├── forecasting.py
│   ├── optimization.py
│   └── kpi_calculation.py
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

```bash
pip install -r requirements.txt
jupyter lab
```
Open the notebook:
```bash
notebooks/01_last_mile_forecasting_and_optimization.ipynb
```

-----

## Business Relevance

This project mirrors real-world last-mile logistics systems used by large retailers and marketplaces. The architecture separates prediction from decision-making, enabling scalable experimentation and operational optimization across regions and demand scenarios.
