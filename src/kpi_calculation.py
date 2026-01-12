def compute_kpis(results_df, forecast_df):
    """
    Compute key operational KPIs from optimization results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of the optimization model
    forecast_df : pd.DataFrame
        Demand forecast data

    Returns
    -------
    kpis : dict
        Dictionary of high-level performance metrics
    """

    # Total forecasted demand across zones and hours
    total_forecast = forecast_df["forecast_orders"].sum()

    # Total late orders implied by optimization
    total_late = results_df["late_orders"].sum()

    # Total driver-hours allocated
    total_drivers = results_df["drivers"].sum()

    kpis = {
        # Cost is set externally to allow flexible objective definitions
        "total_cost": None,
        "late_orders": total_late,
        "driver_hours": total_drivers,
        "on_time_rate": 1 - (total_late / total_forecast)
    }

    return kpis
