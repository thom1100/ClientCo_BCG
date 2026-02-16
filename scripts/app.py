
import streamlit as st
import pandas as pd
import plotly.express as px
import json

@st.cache_data
def load_data():

    yield_df = pd.read_csv("../data/barley_yield_from_1982.csv", sep=";")
    departement_df = pd.read_csv("../data/departements-region.csv")

    departement_df["department"] = (
        departement_df["department"]
        .str.replace("-", "_")
        .str.replace("é", "e")
        .str.replace("è","e")
        .str.replace("ô", "o")
        .str.replace("'","_")
        .str.replace(" ", "_")
        .str.replace("_Py", "_py")
    )

    yield_df = yield_df.merge(departement_df, on="department", how="left")
    yield_df = yield_df[yield_df["department"]!="Seine_SeineOise"]

    yield_df["production"] = yield_df["production"].interpolate()
    yield_df["area"] = yield_df["area"].interpolate()
    yield_df["yield"] = yield_df["production"]/yield_df["area"]

    return yield_df
@st.cache_data
def load_weather():

    weather_df = pd.read_parquet("../data/climate_data_from_1982.parquet")

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

    return weather_yearly

weather_yearly = load_weather()
yield_df = load_data()

st.title("🌾 French Barley Productivity Dashboard")

tab1, tab2 = st.tabs([
    "Barley Yield Data",
    "Weather Data"
])

with tab1:
    with st.sidebar:
        st.sidebar.title("Filters")


    st.header("Yield Analysis")

    year_range = st.sidebar.slider(
        "Select year range",
        int(yield_df.year.min()),
        int(yield_df.year.max()),
        (2000, 2015)
    )

    target = st.sidebar.radio(
        "Select target",
        ["yield", "production"]
    )


    filtered = yield_df[
        (yield_df.year >= year_range[0]) &
        (yield_df.year <= year_range[1])
    ]

    st.subheader(f"{target.capitalize()} evolution across departments")

    map_anim_df = (
        filtered
        .groupby(["year", "department"])
        .agg(target_mean=(target, "mean"))
        .reset_index()
    )


    with open("../data/departements.geojson") as f:
        france_geo = json.load(f)

    fig_anim = px.choropleth(
        map_anim_df,
        geojson=france_geo,
        locations="department",
        featureidkey="properties.nom",
        color="target_mean",
        animation_frame="year",
        projection="mercator",
        color_continuous_scale="Viridis"
    )

    fig_anim.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig_anim.update_layout(
        title=f"{target.capitalize()} across French Departments over Time",
        coloraxis=dict(
            cmin=map_anim_df.target_mean.min(),
            cmax=map_anim_df.target_mean.max()
        )
    )
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 300

    st.plotly_chart(
        fig_anim,
        use_container_width=True
    )

    st.subheader("Regional Production Breakdown")

    yield_agg = (
        filtered
        .groupby(["region", "department"])
        .agg(
            yield_mean=("yield", "mean"),
            production_mean=("production", "mean"),
            area_mean=("area", "mean")
        )
        .reset_index()
    )

    fig_tree = px.treemap(
        yield_agg,
        path=['region', 'department'],
        values='production_mean',
        color='yield_mean',
        hover_data=['area_mean'],
        title="Production Distribution by Region and Department"
    )

    st.plotly_chart(fig_tree, use_container_width=True)

    st.subheader(f"Yearly {target.capitalize()} Trend per Region")

    region_df = (
        filtered
        .groupby(["year","region"])
        .agg(target_mean=(target,"mean"))
        .reset_index()
    )

    fig_line = px.line(
        region_df,
        x="year",
        y="target_mean",
        color="region",
        markers=True,
        title=f"{target.capitalize()} Evolution per Region"
    )

    st.plotly_chart(fig_line, use_container_width=True)



    regions = st.sidebar.multiselect(
        "Select region",
        yield_df.region.unique(),
        default=yield_df.region.unique()
    )

    filtered = filtered[filtered.region.isin(regions)]

    region_df["yoy"] = (
        region_df
        .groupby("region")["target_mean"]
        .pct_change()*100
    )

with tab2:
    st.header("🌦️ Weather Data Analysis")

    st.subheader("Climate evolution across departments")

    col1, col2, col3 = st.columns(3)

    with col1:
        scenario = st.selectbox(
            "Scenario",
            weather_yearly.scenario.unique()
        )

    with col2:
        department = st.selectbox(
            "Department",
            weather_yearly.nom_dep.unique()
        )

    with col3:
        metric = st.selectbox(
            "Metric",
            [
                "mean_temp",
                "mean_max_temp",
                "total_precip"
            ]
        )

        weather_filtered = weather_yearly[
    (weather_yearly.scenario == scenario) &
    (weather_yearly.nom_dep == department)
    ]

    fig_weather = px.line(
    weather_filtered,
    x="year",
    y=metric,
    markers=True,
    title=f"{metric} evolution in {department} ({scenario})"
)

    st.plotly_chart(fig_weather, use_container_width=True)

    st.subheader("Compare departments")

    multi_dep = st.multiselect(
        "Select departments",
        weather_yearly.nom_dep.unique(),
        default=[department]
    )

    weather_multi = weather_yearly[
        (weather_yearly.scenario == scenario) &
        (weather_yearly.nom_dep.isin(multi_dep))
    ]

    fig_multi = px.line(
        weather_multi,
        x="year",
        y=metric,
        color="nom_dep",
        title=f"{metric} comparison across departments"
    )

    st.plotly_chart(fig_multi, use_container_width=True)

    st.subheader("Climate Metrics Correlation")

    corr = weather_yearly[
        [
            "mean_temp",
            "std_temp",
            "mean_max_temp",
            "std_max_temp",
            "total_precip",
            "std_precip"
        ]
    ].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        title="Correlation between climate indicators"
    )

    st.plotly_chart(fig_corr, use_container_width=True)









