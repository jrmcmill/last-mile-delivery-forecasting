import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def prepare_hourly_demand(orders_df):
    """
    Aggregate order-level data into hourly demand by zone.

    This creates the modeling dataset used for demand forecasting.

    Parameters
    ----------
    orders_df : pd.DataFrame
        Order-level delivery dataset

    Returns
    -------
    hourly : pd.DataFrame
        Hourly demand with engineered time-based features
    """

    # Aggregate order counts by hour and zone
    hourly = (
        orders_df
        .groupby([pd.Grouper(key="timestamp", freq="h"), "zone_id"])
        .size()
        .reset_index(name="orders")
    )

    # Time-based features commonly used in demand forecasting
    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["dayofweek"] = hourly["timestamp"].dt.dayofweek
    hourly["is_weekend"] = hourly["dayofweek"] >= 5

    return hourly


def train_demand_model(hourly_demand):
    """
    Train a machine learning model to forecast hourly delivery demand.

    Uses gradient boosting to capture non-linear time and zone effects.

    Parameters
    ----------
    hourly_demand : pd.DataFrame
        Hourly demand dataset with features

    Returns
    -------
    model : GradientBoostingRegressor
        Trained demand forecasting model
    mae : float
        Mean Absolute Error on holdout data
    """

    # Feature matrix and target variable
    X = hourly_demand[["hour", "dayofweek", "is_weekend", "zone_id"]]
    y = hourly_demand["orders"]

    # Time-aware train/test split (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False
    )

    # Train gradient boosting regression model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Evaluate forecast accuracy using MAE
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    return model, mae


def forecast_demand(model, zones, periods=168, start_timestamp=None):
    """
    Generate future demand forecasts by zone and hour.

    Parameters
    ----------
    model : trained regression model
        Demand forecasting model
    zones : list
        List of zone identifiers
    periods : int
        Number of future hours to forecast
    start_timestamp : pd.Timestamp
        Forecast start time

    Returns
    -------
    future_df : pd.DataFrame
        Hourly demand forecasts by zone
    """

    if start_timestamp is None:
        raise ValueError("start_timestamp must be provided")

    # Generate future hourly timestamps
    future = pd.date_range(
        start=start_timestamp,
        periods=periods,
        freq="h"
    )

    # Create full zone × time forecast grid
    future_df = pd.DataFrame({
        "timestamp": future.repeat(len(zones)),
        "zone_id": list(zones) * len(future)
    })

    # Feature engineering consistent with training data
    future_df["hour"] = future_df["timestamp"].dt.hour
    future_df["dayofweek"] = future_df["timestamp"].dt.dayofweek
    future_df["is_weekend"] = future_df["dayofweek"] >= 5

    # Generate demand forecasts
    future_df["forecast_orders"] = model.predict(
        future_df[["hour", "dayofweek", "is_weekend", "zone_id"]]
    )

    return future_df
