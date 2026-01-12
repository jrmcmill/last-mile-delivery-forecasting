import numpy as np
import pandas as pd


def generate_synthetic_orders(
    start_date="2024-01-01",
    end_date="2024-03-01",
    zones=5,
    seed=42
):
    """
    Generate a synthetic last-mile delivery order dataset.

    The dataset simulates hourly delivery demand across multiple zones,
    incorporating realistic rush-hour demand spikes and stochastic
    delivery service times.

    Parameters
    ----------
    start_date : str
        Start date for simulation (inclusive)
    end_date : str
        End date for simulation (inclusive)
    zones : int
        Number of delivery zones
    seed : int
        Random seed for reproducibility

    Returns
    -------
    orders_df : pd.DataFrame
        Order-level dataset with timestamps, zones, delivery times,
        and SLA indicators
    """

    # Ensure reproducibility for experimentation and modeling
    np.random.seed(seed)

    # Generate an hourly time index for the simulation window
    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq="h"
    )

    orders = []

    # Iterate over each hour in the simulation period
    for ts in date_range:
        hour = ts.hour

        # Flag rush-hour periods with higher congestion and demand
        is_rush = hour in [7, 8, 9, 16, 17, 18]

        # Simulate demand independently for each delivery zone
        for zone_id in range(zones):

            # Baseline Poisson demand for normal hours
            base_demand = np.random.poisson(3)

            # Add incremental demand during rush hours
            if is_rush:
                base_demand += np.random.poisson(4)

            # Generate individual orders for the hour/zone
            for _ in range(base_demand):
                orders.append({
                    "timestamp": ts,
                    "zone_id": zone_id,
                    "order_id": f"O{np.random.randint(1e9)}",
                    "delivery_time_min": simulate_delivery_time(hour)
                })

    # Convert to DataFrame for downstream modeling
    orders_df = pd.DataFrame(orders)

    # SLA flag: on-time delivery defined as <= 45 minutes
    orders_df["on_time"] = (orders_df["delivery_time_min"] <= 45).astype(int)

    return orders_df


def simulate_delivery_time(hour):
    """
    Simulate delivery service time in minutes.

    Delivery times increase during rush hours to reflect
    traffic congestion and operational delays.

    Parameters
    ----------
    hour : int
        Hour of day (0–23)

    Returns
    -------
    float
        Simulated delivery time in minutes
    """

    # Base delivery time with random noise
    base = np.random.normal(25, 5)

    # Inflate service times during rush-hour congestion
    if hour in [7, 8, 9, 16, 17, 18]:
        base *= 1.4

    # Enforce a minimum feasible delivery time
    return max(10, base)
