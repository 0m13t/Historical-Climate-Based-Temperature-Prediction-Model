import io
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


#functies

def get_coordinates(city):
    city = city.strip()
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
    data = requests.get(url).json()

    if "results" not in data:
        print("City not found")
        return None, None, None

    result = data["results"][0]

    lat = result["latitude"]
    lon = result["longitude"]
    elevation = result.get("elevation", 0)

    return lat, lon, elevation


def get_weather_data(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "1985-01-01",
        "end_date": "2025-12-31",
        "daily": ["temperature_2m_max", "precipitation_sum", "pressure_msl_mean"],
        "timezone": "auto",
    }

    data = requests.get(url, params=params).json()
    daily = data["daily"]

    df = pd.DataFrame({
        "Date": daily["time"],
        "Max_Temp": daily["temperature_2m_max"],
        "Precipitation": daily["precipitation_sum"],
        "Pressure": daily["pressure_msl_mean"]
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df


def get_nao_data():
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
    res = requests.get(url)

    nao = pd.read_csv(
        io.StringIO(res.text),
        sep=r"\s+",
        header=None,
        names=["Year", "Month", "NAO_Index"]
    )

    nao["Year"] = nao["Year"].astype(int)
    nao["Month"] = nao["Month"].astype(int)

    return nao[nao["Year"] <= 2025]


def add_extra_features(df, lat, lon, elevation):
    df = df.copy()

    df["Latitude"] = lat
    df["Longitude"] = lon
    df["Elevation"] = elevation

    min_year = df["Year"].min()
    df["Year_Trend"] = df["Year"] - min_year

    # season pattern, because day 365 and day 1 are close to each other
    df["Day_Sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["Day_Cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)

    # interaction with location
    df["NAO_Lat"] = df["NAO_Index"] * df["Latitude"]
    df["NAO_Lon"] = df["NAO_Index"] * df["Longitude"]

    return df


def add_baselines(df):
    df = df.copy()

    # avg temp
    normal_temp = (
        df.groupby(["Month", "Day"])["Max_Temp"]
        .mean()
        .reset_index()
        .rename(columns={"Max_Temp": "Climate_Baseline"})
    )

    df = pd.merge(df, normal_temp, on=["Month", "Day"], how="left")

    # last 10 years
    last_year = df["Year"].max()
    recent = df[df["Year"] >= last_year - 9]

    recent_temp = (
        recent.groupby(["Month", "Day"])["Max_Temp"]
        .mean()
        .reset_index()
        .rename(columns={"Max_Temp": "Recent_10yr_Baseline"})
    )

    df = pd.merge(df, recent_temp, on=["Month", "Day"], how="left")
    df["Recent_10yr_Baseline"] = df["Recent_10yr_Baseline"].fillna(df["Climate_Baseline"])

    # Difference from normal
    df["Temp_Anomaly"] = df["Max_Temp"] - df["Climate_Baseline"]

    return df


def make_model():
    model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    return model


def make_range_model(alpha):
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    return model


def estimate_nao(df, month):
    month_data = df[df["Month"] == month]
    return month_data["NAO_Index"].tail(10).mean()


def anomaly_risk(df, month, day, predicted_anomaly):
    day_data = df[(df["Month"] == month) & (df["Day"] == day)]
    anomalies = day_data["Temp_Anomaly"].dropna()

    if len(anomalies) < 10:
        return None

    cold_limit = anomalies.quantile(0.10)
    hot_limit = anomalies.quantile(0.90)
    very_hot_limit = anomalies.quantile(0.95)

    colder_chance = (anomalies <= predicted_anomaly).mean() * 100
    hotter_chance = (anomalies >= predicted_anomaly).mean() * 100

    if predicted_anomaly <= cold_limit:
        category = "Extra cold"
    elif predicted_anomaly >= very_hot_limit:
        category = "Extremely hot"
    elif predicted_anomaly >= hot_limit:
        category = "Extra hot"
    else:
        category = "Normal"

    return {
        "category": category,
        "cold_limit": cold_limit,
        "hot_limit": hot_limit,
        "very_hot_limit": very_hot_limit,
        "colder_chance": colder_chance,
        "hotter_chance": hotter_chance
    }


# mainwork

city = input("Enter city: ")
lat, lon, elevation = get_coordinates(city)

if lat is not None:
    weather = get_weather_data(lat, lon)
    nao = get_nao_data()

    df = pd.merge(weather, nao, on=["Year", "Month"], how="inner")

    df = add_extra_features(df, lat, lon, elevation)
    df = add_baselines(df)

    print("\nCorrelations with Max Temp")
    corr = df[
        [
            "Max_Temp",
            "Pressure",
            "NAO_Index",
            "Climate_Baseline",
            "Recent_10yr_Baseline",
            "Year_Trend"
        ]
    ].corr()["Max_Temp"]

    print(f"Pressure: {corr['Pressure'] * 100:.1f}%")
    print(f"NAO Index: {corr['NAO_Index'] * 100:.1f}%")
    print(f"Climate Baseline: {corr['Climate_Baseline'] * 100:.1f}%")
    print(f"Recent 10yr Baseline: {corr['Recent_10yr_Baseline'] * 100:.1f}%")
    print(f"Year Trend: {corr['Year_Trend'] * 100:.1f}%")

    train = df["Year"] <= 2023
    test = df["Year"] > 2023

    train_years = df.loc[train, "Year"]

    # newer years get a bit more weight
    min_weight = 0.50
    weights = min_weight + (1 - min_weight) * (
        (train_years - train_years.min()) / (train_years.max() - train_years.min())
    )

    features = [
        "Month",
        "Day",
        "DayOfYear",
        "Day_Sin",
        "Day_Cos",
        "Latitude",
        "Longitude",
        "Elevation",
        "Year_Trend",
        "Climate_Baseline",
        "Recent_10yr_Baseline",
        "Pressure",
        "NAO_Index",
        "NAO_Lat",
        "NAO_Lon"
    ]

    X = df[features]
    y = df["Temp_Anomaly"]
    y_real_temp = df["Max_Temp"]

    model = make_model()
    model.fit(X[train], y[train], sample_weight=weights)

    test_anomaly = model.predict(X[test])
    test_temp = df.loc[test, "Climate_Baseline"].values + test_anomaly

    mae = mean_absolute_error(y_real_temp[test], test_temp)

    print(f"\nModel Error (MAE): {mae:.2f}°C")

    low_model = make_range_model(0.10)
    mid_model = make_range_model(0.50)
    high_model = make_range_model(0.90)

    low_model.fit(X[train], y[train], sample_weight=weights)
    mid_model.fit(X[train], y[train], sample_weight=weights)
    high_model.fit(X[train], y[train], sample_weight=weights)

    print("\n--- Future Scenario ---")
    year = int(input("Enter target year: "))
    month = int(input("Enter month (1-12): "))
    day = int(input("Enter day (1-31): "))

    date = pd.Timestamp(year=year, month=month, day=day)
    day_of_year = date.dayofyear

    same_day = df[(df["Month"] == month) & (df["Day"] == day)]

    climate_baseline = same_day["Max_Temp"].mean()

    recent_baseline = same_day[
        same_day["Year"] >= df["Year"].max() - 9
    ]["Max_Temp"].mean()

    if np.isnan(recent_baseline):
        recent_baseline = climate_baseline

    pressure = same_day["Pressure"].mean()
    nao_value = estimate_nao(df, month)

    min_year = df["Year"].min()

    future_row = {
        "Month": month,
        "Day": day,
        "DayOfYear": day_of_year,
        "Day_Sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "Day_Cos": np.cos(2 * np.pi * day_of_year / 365.25),

        "Latitude": lat,
        "Longitude": lon,
        "Elevation": elevation,

        "Year_Trend": year - min_year,

        "Climate_Baseline": climate_baseline,
        "Recent_10yr_Baseline": recent_baseline,

        "Pressure": pressure,

        "NAO_Index": nao_value,
        "NAO_Lat": nao_value * lat,
        "NAO_Lon": nao_value * lon
    }

    future = pd.DataFrame([future_row])
    future = future[features]

    anomaly_low = low_model.predict(future)[0]
    anomaly_mid = mid_model.predict(future)[0]
    anomaly_high = high_model.predict(future)[0]

    temp_low = climate_baseline + anomaly_low
    temp_mid = climate_baseline + anomaly_mid
    temp_high = climate_baseline + anomaly_high

    risk = anomaly_risk(df, month, day, anomaly_mid)

    print(f"\nPrediction for {city} on {day}-{month}-{year}")
    print(f"Latitude: {lat:.2f}")
    print(f"Longitude: {lon:.2f}")
    print(f"Elevation: {elevation:.1f} m")
    print(f"Climate baseline: {climate_baseline:.1f}°C")
    print(f"Recent 10 year baseline: {recent_baseline:.1f}°C")
    print(f"Estimated pressure: {pressure:.1f} hPa")
    print(f"Estimated NAO index: {nao_value:.2f}")

    print(f"\nPredicted anomaly: {anomaly_mid:+.1f}°C")
    print(f"Likely maximum temperature: {temp_mid:.1f}°C")
    print(f"Expected range: {temp_low:.1f}°C to {temp_high:.1f}°C")

    print("\n--- Anomaly check ---")

    if risk is not None:
        print(f"Category: {risk['category']}")
        print(f"Extra cold limit: {risk['cold_limit']:+.1f}°C anomaly")
        print(f"Extra hot limit: {risk['hot_limit']:+.1f}°C anomaly")
        print(f"Very hot limit: {risk['very_hot_limit']:+.1f}°C anomaly")
        print(f"Historical chance of being this cold or colder: {risk['colder_chance']:.1f}%")
        print(f"Historical chance of being this hot or hotter: {risk['hotter_chance']:.1f}%")
    else:
        print("Not enough data for anomaly check")
