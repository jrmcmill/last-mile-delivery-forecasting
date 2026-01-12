from ortools.linear_solver import pywraplp
import pandas as pd


def optimize_driver_allocation(
    forecast_df,
    cost_per_driver=30,
    late_penalty=50,
    capacity_per_driver=4,
    hours_to_optimize=24
):
    """
    Optimize driver allocation to minimize cost and late deliveries.

    This function formulates and solves a linear optimization problem
    where drivers are allocated by zone and hour to meet forecasted demand.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Hourly demand forecasts
    cost_per_driver : float
        Cost per driver per hour
    late_penalty : float
        Penalty cost per late order
    capacity_per_driver : int
        Orders that a single driver can handle per hour
    hours_to_optimize : int
        Number of hours to include in optimization window

    Returns
    -------
    results_df : pd.DataFrame
        Optimized driver allocations and late orders
    total_cost : float
        Objective function value
    """

    # Initialize OR-Tools solver
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Restrict optimization horizon for tractability
    hours = forecast_df["timestamp"].unique()[:hours_to_optimize]
    zones = forecast_df["zone_id"].unique()

    drivers = {}
    late_orders = {}

    # Decision variables: drivers and late orders per zone/hour
    for h in hours:
        for z in zones:
            drivers[h, z] = solver.IntVar(0, solver.infinity(), f"d_{h}_{z}")
            late_orders[h, z] = solver.IntVar(0, solver.infinity(), f"l_{h}_{z}")

    # Demand satisfaction constraints
    for h in hours:
        for z in zones:
            demand = forecast_df[
                (forecast_df["timestamp"] == h) &
                (forecast_df["zone_id"] == z)
            ]["forecast_orders"].values[0]

            solver.Add(
                drivers[h, z] * capacity_per_driver + late_orders[h, z] >= demand
            )

    # Objective: minimize labor cost and lateness penalties
    objective = solver.Objective()
    for h, z in drivers:
        objective.SetCoefficient(drivers[h, z], cost_per_driver)
        objective.SetCoefficient(late_orders[h, z], late_penalty)

    objective.SetMinimization()

    # Solve optimization problem
    solver.Solve()

    # Extract solution into DataFrame
    results = []
    for h, z in drivers:
        results.append({
            "timestamp": h,
            "zone_id": z,
            "drivers": drivers[h, z].solution_value(),
            "late_orders": late_orders[h, z].solution_value()
        })

    results_df = pd.DataFrame(results)

    return results_df, objective.Value()
