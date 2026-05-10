
import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder



BIKE_SHARING_ZIP  = "data/bike-sharing-demand.zip"   
POPULATION_ZIP    = "data/archive (4).zip"        
ALFA_PROCESSED_ZIP = "data/processed.zip"          

EXTRACT_DIR = "extracted_data"                       
RANDOM_SEED = 42


def extract_zip(zip_path: str, dest_dir: str) -> str:
    """Extract a zip archive and return the destination directory."""
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"   Extracted '{zip_path}' → '{dest_dir}'")
    return dest_dir



# 1. DEMAND FORECASTING — Bike Sharing Dataset

def load_bike_sharing(zip_path: str = BIKE_SHARING_ZIP) -> dict:
    """
    Loads the Kaggle Bike Sharing Demand dataset.

    Columns (train):
        datetime, season, holiday, workingday, weather,
        temp, atemp, humidity, windspeed, casual, registered, count

    Returns:
        dict with keys 'train', 'test'
    """
    print("\n[1/4] Loading Bike Sharing Demand dataset …")
    dest = os.path.join(EXTRACT_DIR, "bike_sharing")
    extract_zip(zip_path, dest)

    train_path = os.path.join(dest, "train.csv")
    test_path  = os.path.join(dest, "test.csv")

    train = pd.read_csv(train_path, parse_dates=["datetime"])
    test  = pd.read_csv(test_path,  parse_dates=["datetime"])

    #  Feature engineering 
    for df in (train, test):
        df["hour"]       = df["datetime"].dt.hour
        df["day_of_week"]= df["datetime"].dt.dayofweek
        df["month"]      = df["datetime"].dt.month
        df["year"]       = df["datetime"].dt.year

    # Season / weather labels (for readability)
    season_map  = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_map = {1: "Clear", 2: "Mist", 3: "Light Snow/Rain", 4: "Heavy Rain"}
    for df in (train, test):
        df["season_label"]  = df["season"].map(season_map)
        df["weather_label"] = df["weather"].map(weather_map)

    print(f"  Train shape : {train.shape}")
    print(f"  Test  shape : {test.shape}")
    print(f"  Date range  : {train['datetime'].min().date()} → {train['datetime'].max().date()}")
    print(f"  Target col  : 'count'  |  min={train['count'].min()}, max={train['count'].max()}")
    return {"train": train, "test": test}



# 2. POPULATION DENSITY — US City Population Density

def load_population_density(zip_path: str = POPULATION_ZIP) -> pd.DataFrame:
    """
    Loads the US City Population Density dataset.

    Columns:
        Index, City, State,
        Population Density (Persons/Square Mile),
        2016 Population, Land Area (Square Miles)

    Returns:
        pd.DataFrame
    """
    print("\n[2/4] Loading Population Density dataset …")
    dest = os.path.join(EXTRACT_DIR, "population_density")
    extract_zip(zip_path, dest)

    csv_path = os.path.join(dest, "uscitypopdensity.csv")
    df = pd.read_csv(csv_path)

    # Rename for convenience
    df.rename(columns={
        "Population Density (Persons/Square Mile)": "pop_density",
        "2016 Population":                          "population_2016",
        "Land Area (Square Miles)":                 "land_area_sqmi"
    }, inplace=True)

    # Density tiers (useful for routing/coverage analysis)
    bins   = [0, 1_000, 5_000, 15_000, float("inf")]
    labels = ["Low", "Medium", "High", "Very High"]
    df["density_tier"] = pd.cut(df["pop_density"], bins=bins, labels=labels)

    print(f"  Shape   : {df.shape}")
    print(f"  States  : {df['State'].nunique()}")
    print(f"  Density range: {df['pop_density'].min():,.0f} – {df['pop_density'].max():,.0f} persons/sq mi")
    return df



def load_drone_telemetry(
    processed_zip: str = ALFA_PROCESSED_ZIP,
    n_synthetic: int = 5_000
) -> pd.DataFrame:
    """
    Loads the ALFA UAV dataset directly from processed.zip.

    Each flight folder inside the zip contains multiple CSVs per sensor topic.
    We load the most useful ones per flight and merge them:
        - mavctrl-rpy.csv       → roll, pitch, yaw
        - mavros-battery.csv    → battery voltage
        - mavros-global_position-local.csv → position (x, y, z)

    The fault_type label is extracted from the flight folder name
    e.g. "carbonZ_..._engine_failure" → "engine_failure"
         "carbonZ_..._no_failure"     → "no_failure"

    Returns:
        pd.DataFrame with columns:
            flight_id, fault_type, timestamp_ns,
            roll, pitch, yaw,
            battery_voltage, battery_pct,
            pos_x, pos_y, pos_z
    """
    print("\n[3/4] Loading ALFA Drone Telemetry dataset …")

    if not os.path.exists(processed_zip):
        print(f"  ⚠ '{processed_zip}' not found — generating synthetic telemetry data.")
        return _synthetic_drone_telemetry(n_synthetic)

    flights = []

    with zipfile.ZipFile(processed_zip, "r") as zf:
        all_names = zf.namelist()

        # Collect unique flight folder names
        flight_folders = sorted(set(
            n.split("/")[1] for n in all_names
            if n.startswith("processed/") and n.count("/") >= 2 and n.split("/")[1]
        ))

        print(f"  Found {len(flight_folders)} flight sequences …")

        for folder in flight_folders:
            # Extract fault label from folder name
            parts = folder.split("_")
            
          
            try:
                label_parts = parts[2:]  
                # join and clean up
                fault_type = "_".join(label_parts).replace("__", "_").strip("_")
                if not fault_type:
                    fault_type = "unknown"
            except IndexError:
                fault_type = "unknown"

            prefix = f"processed/{folder}/{folder}"

            # ── Roll / Pitch / Yaw 
            rpy_file = f"{prefix}-mavctrl-rpy.csv"
            # ── Battery 
            bat_file = f"{prefix}-mavros-battery.csv"
            # ── Local position 
            pos_file = f"{prefix}-mavros-global_position-local.csv"

            try:
                with zf.open(rpy_file) as f:
                    rpy = pd.read_csv(f, usecols=["%time", "field.x", "field.y", "field.z"])
                    rpy.rename(columns={
                        "%time":   "timestamp_ns",
                        "field.x": "roll",
                        "field.y": "pitch",
                        "field.z": "yaw"
                    }, inplace=True)
            except KeyError:
                continue  # skip flight if key CSV missing

            try:
                with zf.open(bat_file) as f:
                    bat = pd.read_csv(f, usecols=["%time", "field.voltage", "field.percentage"])
                    bat.rename(columns={
                        "%time":            "timestamp_ns",
                        "field.voltage":    "battery_voltage",
                        "field.percentage": "battery_pct"
                    }, inplace=True)
            except KeyError:
                bat = pd.DataFrame(columns=["timestamp_ns","battery_voltage","battery_pct"])

            try:
                with zf.open(pos_file) as f:
                    pos = pd.read_csv(f, usecols=[
                        "%time",
                        "field.pose.pose.position.x",
                        "field.pose.pose.position.y",
                        "field.pose.pose.position.z"
                    ])
                    pos.rename(columns={
                        "%time":                          "timestamp_ns",
                        "field.pose.pose.position.x":    "pos_x",
                        "field.pose.pose.position.y":    "pos_y",
                        "field.pose.pose.position.z":    "pos_z"
                    }, inplace=True)
            except KeyError:
                pos = pd.DataFrame(columns=["timestamp_ns","pos_x","pos_y","pos_z"])

            # ── Merge on nearest timestamp 
            rpy = rpy.sort_values("timestamp_ns")
            if not bat.empty:
                bat = bat.sort_values("timestamp_ns")
                merged = pd.merge_asof(rpy, bat, on="timestamp_ns", direction="nearest")
            else:
                merged = rpy.copy()
                merged["battery_voltage"] = float("nan")
                merged["battery_pct"]     = float("nan")

            if not pos.empty:
                pos = pos.sort_values("timestamp_ns")
                merged = pd.merge_asof(merged, pos, on="timestamp_ns", direction="nearest")
            else:
                merged["pos_x"] = float("nan")
                merged["pos_y"] = float("nan")
                merged["pos_z"] = float("nan")

            merged["flight_id"]  = folder
            merged["fault_type"] = fault_type
            flights.append(merged)

    if not flights:
        print("   No valid flight CSVs found — falling back to synthetic data.")
        return _synthetic_drone_telemetry(n_synthetic)

    df = pd.concat(flights, ignore_index=True)

    # Encode fault label numerically
    le = LabelEncoder()
    df["fault_label"] = le.fit_transform(df["fault_type"])

    print(f"   ALFA dataset loaded  shape={df.shape}")
    print(f"  Fault distribution:\n{df['fault_type'].value_counts().to_string()}")
    return df


def _synthetic_drone_telemetry(n: int = 5_000) -> pd.DataFrame:
    """Synthetic fallback — realistic drone telemetry with injected faults."""
    np.random.seed(RANDOM_SEED)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1s")
    flight_ids = np.repeat(np.arange(1, 11), n // 10)[:n]

    roll    = np.random.normal(0, 5,  n)
    pitch   = np.random.normal(0, 5,  n)
    yaw     = np.random.uniform(0, 360, n)
    pos_x   = np.cumsum(np.random.normal(0, 1, n))
    pos_y   = np.cumsum(np.random.normal(0, 1, n))
    pos_z   = np.random.normal(100, 20, n).clip(0, 400)
    voltage = np.random.normal(12.6, 0.5, n).clip(9, 16.8)
    bat_pct = np.random.uniform(0.2, 1.0, n)

    fault_types = np.random.choice(
        ["no_failure", "engine_failure", "elevator_failure",
         "rudder_right_failure", "aileron_failure"],
        n, p=[0.70, 0.12, 0.08, 0.05, 0.05]
    )

    eng  = fault_types == "engine_failure"
    elev = fault_types == "elevator_failure"
    roll[eng]    += np.random.normal(0, 30, eng.sum())
    pitch[elev]  += np.random.normal(0, 40, elev.sum())
    voltage[eng]  = np.random.uniform(9.0, 10.5, eng.sum())

    df = pd.DataFrame({
        "timestamp_ns":    pd.to_datetime(timestamps).astype(np.int64),
        "flight_id":       flight_ids,
        "roll":            roll,
        "pitch":           pitch,
        "yaw":             yaw,
        "pos_x":           pos_x,
        "pos_y":           pos_y,
        "pos_z":           pos_z,
        "battery_voltage": voltage,
        "battery_pct":     bat_pct,
        "fault_type":      fault_types,
    })
    le = LabelEncoder()
    df["fault_label"] = le.fit_transform(df["fault_type"])
    print(f"   Synthetic telemetry generated  shape={df.shape}")
    print(f"  Fault distribution:\n{df['fault_type'].value_counts().to_string()}")
    return df



# 4 ANOMALY DETECTION — Synthetic Data

def load_anomaly_data(n_normal: int = 4_000, n_anomaly: int = 400) -> pd.DataFrame:
    """
    Generates a synthetic tabular dataset for anomaly detection.

    Simulates sensor readings from a drone delivery network hub.
    Normal samples follow multivariate Gaussian distributions;
    anomalies are injected as out-of-distribution points.

    Columns:
        delivery_time_min, battery_drain_pct, route_deviation_km,
        wind_speed_kmh, payload_weight_kg, signal_strength_dbm,
        is_anomaly  (0 = normal, 1 = anomaly)

    Returns:
        pd.DataFrame
    """
    print("\n[4/4] Generating Anomaly Detection dataset …")
    np.random.seed(RANDOM_SEED)

    
    normal = pd.DataFrame({
        "delivery_time_min":    np.random.normal(25,   4,    n_normal),
        "battery_drain_pct":    np.random.normal(30,   5,    n_normal),
        "route_deviation_km":   np.abs(np.random.normal(0, 0.5, n_normal)),
        "wind_speed_kmh":       np.random.normal(15,   5,    n_normal).clip(0, 60),
        "payload_weight_kg":    np.random.normal(2.0,  0.4,  n_normal).clip(0.5, 5),
        "signal_strength_dbm":  np.random.normal(-65,  8,    n_normal),
        "is_anomaly":           0,
    })

   
    n_each = n_anomaly // 3

    type_a = pd.DataFrame({
        "delivery_time_min":   np.random.normal(60,  10, n_each),
        "battery_drain_pct":   np.random.normal(75,  10, n_each),
        "route_deviation_km":  np.random.normal(5,   1,  n_each),
        "wind_speed_kmh":      np.random.normal(15,  5,  n_each).clip(0, 60),
        "payload_weight_kg":   np.random.normal(2.0, 0.4,n_each).clip(0.5, 5),
        "signal_strength_dbm": np.random.normal(-65, 8,  n_each),
        "is_anomaly":          1,
    })

    # Type B: critical battery drop with poor signal (hardware fault)
    type_b = pd.DataFrame({
        "delivery_time_min":   np.random.normal(28,  5,  n_each),
        "battery_drain_pct":   np.random.normal(95,  3,  n_each),
        "route_deviation_km":  np.abs(np.random.normal(0, 0.5, n_each)),
        "wind_speed_kmh":      np.random.normal(15,  5,  n_each).clip(0, 60),
        "payload_weight_kg":   np.random.normal(2.0, 0.4,n_each).clip(0.5, 5),
        "signal_strength_dbm": np.random.normal(-95, 5,  n_each),
        "is_anomaly":          1,
    })

    # Type C: extreme wind causing payload overload scenario
    remaining = n_anomaly - 2 * n_each
    type_c = pd.DataFrame({
        "delivery_time_min":   np.random.normal(45,  8,  remaining),
        "battery_drain_pct":   np.random.normal(55,  10, remaining),
        "route_deviation_km":  np.random.normal(3,   1,  remaining),
        "wind_speed_kmh":      np.random.normal(55,  5,  remaining).clip(0, 80),
        "payload_weight_kg":   np.random.normal(4.5, 0.3,remaining).clip(0.5, 5),
        "signal_strength_dbm": np.random.normal(-70, 8,  remaining),
        "is_anomaly":          1,
    })

    df = pd.concat([normal, type_a, type_b, type_c], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"   Anomaly dataset generated  shape={df.shape}")
    print(f"  Normal : {(df['is_anomaly']==0).sum()}")
    print(f"  Anomaly: {(df['is_anomaly']==1).sum()}")
    return df



# MAIN — load everything and print a summary

if __name__ == "__main__":
    print("=" * 60)
    print("  BSDS SP2026 — Dataset Loading Script")
    print("=" * 60)

    # ── Load all four datasets 
    bike_data    = load_bike_sharing()
    pop_density  = load_population_density()
    drone_telem  = load_drone_telemetry(ALFA_PROCESSED_ZIP)
    anomaly_data = load_anomaly_data()

    # ── Quick summaries 
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print("\n Bike Sharing (train):")
    print(bike_data["train"][["datetime","season_label","weather_label","temp","humidity","count"]].head(3).to_string(index=False))

    print("\n  Population Density:")
    print(pop_density[["City","State","pop_density","density_tier"]].head(3).to_string(index=False))

    print("\n Drone Telemetry:")
    print(drone_telem[["flight_id","roll","pitch","yaw","battery_voltage","fault_type"]].head(3).to_string(index=False))

    print("\n Anomaly Detection:")
    print(anomaly_data[["delivery_time_min","battery_drain_pct","route_deviation_km","is_anomaly"]].head(3).to_string(index=False))

    print("\n All datasets loaded successfully!")

   