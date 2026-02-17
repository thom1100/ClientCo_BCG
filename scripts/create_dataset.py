import pandas as pd
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

    weather_yearly_scenario = weather_yearly[(weather_yearly["scenario"]== scenario) | (weather_yearly["scenario"]== "historical")]
    weather_yearly_scenario = weather_yearly_scenario.rename(columns={"nom_dep":"department"})
    barley_df = barley_df[["department", "year", "yield"]]
    df_final = weather_yearly_scenario.merge(barley_df,
                               on=["department", "year"],
                               how="left")

    df_final.to_csv(f"../data/weather_agg_{scenario}.csv", index=False)

if __name__=="__main__":
    weather_yearly("optimistic")