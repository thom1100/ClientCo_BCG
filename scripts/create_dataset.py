import pandas as pd
import numpy as np

def rain_features_one_group(g: pd.DataFrame, rain_threshold_mm: float = 0.0001) -> pd.Series:
    g = g.sort_values("time")
    p = g["precipitation"].astype(float).fillna(0.0)

    rain = p >= rain_threshold_mm

    # run ids whenever rain status changes
    run_id = rain.ne(rain.shift(fill_value=False)).cumsum()

    # lengths of each run
    run_size = rain.groupby(run_id).size()
    run_is_rain = rain.groupby(run_id).first()

    rain_run_lengths = run_size[run_is_rain]  # <-- correct alignment

    max_consec = int(rain_run_lengths.max()) if len(rain_run_lengths) else 0
    min_consec = int(rain_run_lengths.min()) if len(rain_run_lengths) else 0

    max_daily_precip = float(p.max()) if len(p) else np.nan

    return pd.Series({
        "max_consec_rain_days": max_consec,
        "min_consec_rain_days": min_consec,
        "max_daily_precip": max_daily_precip,
        "rainy_days_count": int(rain.sum()),
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
        .groupby(["scenario", "nom_dep", "code_dep", "year"])
        .apply(lambda g: rain_features_one_group(g, rain_threshold_mm=0.00008))
        .reset_index()
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
    weather_yearly("pessimistic")
    weather_yearly("neutral")