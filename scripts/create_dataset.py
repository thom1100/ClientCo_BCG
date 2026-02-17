import pandas as pd

import numpy as np
import pandas as pd

def rain_features_one_group(g: pd.DataFrame, rain_threshold: float = 0.0) -> pd.Series:
    """
    g: daily rows for ONE (scenario, nom_dep, code_dep, year)
       must contain columns: time (datetime64), precipitation (float)
    """
    g = g.sort_values("time")

    p = g["precipitation"].astype(float)
    rain = p > rain_threshold

    # --- consecutive rain runs (lengths) ---
    # Identify run blocks whenever rain/non-rain changes
    run_id = rain.ne(rain.shift(fill_value=False)).cumsum()
    run_len = rain.groupby(run_id).sum()          # sums booleans -> length of rain days in each run block
    run_is_rain = rain.groupby(run_id).first()    # whether that run block is a rain-run

    rain_run_lengths = run_len[run_is_rain.values]  # lengths of only rain runs

    max_consec_rain = int(rain_run_lengths.max()) if len(rain_run_lengths) else 0
    min_consec_rain = int(rain_run_lengths.min()) if len(rain_run_lengths) else 0

    # --- max daily precipitation ---
    max_precip = float(p.max()) if len(p) else np.nan

    # --- average days between 2 rainy days ---
    rain_dates = g.loc[rain, "time"].dt.normalize()
    if len(rain_dates) >= 2:
        avg_days_between_rain = float(rain_dates.diff().dt.days.dropna().mean())
    else:
        avg_days_between_rain = np.nan

    return pd.Series({
        "max_consec_rain_days": max_consec_rain,
        "min_consec_rain_days": min_consec_rain,
        "max_daily_precip": max_precip,
        "avg_days_between_rain_days": avg_days_between_rain
    })


scenarios = {"optimistic":"ssp1_2_6",
             "neutral":"ssp2_4_5",
             "pessimistic":"ssp5_8_5"}

def weather_yearly(scenario):

    weather_df = pd.read_parquet("../data/climate_data_from_1982.parquet")
    barley_df = pd.read_csv("../data/barley_yield_from_1982.csv", sep= ";")
    barley_df["production"]=barley_df["production"].interpolate(method="linear")
    barley_df["area"] = barley_df["area"].interpolate(method="linear")
    barley_df["yield"] = barley_df["production"]/barley_df["area"]
    barley_df["yield"] = barley_df["yield"].fillna(0)

    scenario = scenarios[scenario]
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    weather_pivot = (
        weather_df
        .pivot_table(
            index=["scenario", "nom_dep", "code_dep", "time", "year"],
            columns="metric",
            values="value"
        )
        .reset_index()
    )

    # Yearly rain-run features (needs daily rows, so compute from weather_pivot)
    rain_yearly = (
        weather_pivot
        .groupby(["scenario", "nom_dep", "code_dep", "year"], as_index=False)
        .apply(lambda g: rain_features_one_group(g, rain_threshold=0.0))
        .reset_index(drop=True)
    )


    weather_yearly = weather_pivot.groupby(
        ["scenario","nom_dep","code_dep","year"]
    ).agg(
        mean_temp=("near_surface_air_temperature", "mean"),
        std_temp=("near_surface_air_temperature", "std"),
        mean_max_temp=("daily_maximum_near_surface_air_temperature", "mean"),
        std_max_temp=("daily_maximum_near_surface_air_temperature", "std"),
        total_precip=("precipitation", "sum"),
        std_precip=("precipitation", "std")
    ).reset_index()

    weather_yearly = weather_yearly.merge(
        rain_yearly,
        on=["scenario", "nom_dep", "code_dep", "year"],
        how="left"
    )


    weather_yearly_scenario = weather_yearly[(weather_yearly["scenario"]== scenario) | (weather_yearly["scenario"]== "historical")]
    weather_yearly_scenario = weather_yearly_scenario.rename(columns={"nom_dep":"department"})
    barley_df = barley_df[["department", "year", "yield"]]
    df_final = weather_yearly_scenario.merge(barley_df,
                               on=["department", "year"],
                               how="left")

    df_final.to_csv(f"../data/weather_agg_{scenario}.csv", index=False)

if __name__=="__main__":
    weather_yearly("optimistic")