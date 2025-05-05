import requests
import pandas as pd
import time
import numpy as np


START_YEAR = 2020
END_YEAR = 2021  # 731 days
LOCATIONS = [
    (41.8781, -93.0977), (40.2171, -88.4159), (41.6837, -96.9145), (43.7844, -88.7879), (39.0119, -98.4842),
    (41.2959, -95.8998), (40.6331, -89.3985), (42.4967, -96.4049), (44.3148, -87.8934), (38.5267, -97.6145),
] + [(lat + i * 0.1, lon - i * 0.1) for i in range(15) for lat, lon in [(41.8781, -93.0977)]]  # 25 locations

OUTPUT_FILE = "crop_drought_resistance_suggestion.csv"

def fetch_nasa_data():
    def get_nasa_data(lat, lon):
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M,ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}&start={START_YEAR}0101&end={END_YEAR}1231&format=JSON"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()['properties']['parameter']
            return pd.DataFrame({
                'Date': list(data['PRECTOTCORR'].keys()),
                'Precipitation_mm': [v if v != -999.0 else 0.0 for v in data['PRECTOTCORR'].values()],
                'Temperature_C': list(data['T2M'].values()),
                'Solar_Radiation_MJ_m2': list(data['ALLSKY_SFC_SW_DWN'].values()),
                'Latitude': lat,
                'Longitude': lon
            })
        except Exception as e:
            print(f"NASA error at {lat}, {lon}: {e}")
            return pd.DataFrame()

    print("Fetching NASA data...")
    nasa_dfs = []
    for i, (lat, lon) in enumerate(LOCATIONS, 1):
        print(f"Processing NASA location {i}/{len(LOCATIONS)}...")
        df = get_nasa_data(lat, lon)
        if df.empty and i == 1:
            print("NASA failed on first try. Aborting.")
            return pd.DataFrame()
        nasa_dfs.append(df)
        time.sleep(2)
    return pd.concat(nasa_dfs)

def create_suggestion_dataset():
    nasa_df = fetch_nasa_data()
    if nasa_df.empty:
        print("Failed to fetch NASA data. Exiting.")
        return

    df = nasa_df.copy()
    
    df['Evapotranspiration_mm'] = df['Solar_Radiation_MJ_m2'] * 0.035
    df['Soil_Moisture_%'] = np.clip(df['Precipitation_mm'] * 0.5 - df['Evapotranspiration_mm'] * 0.3, 0, 30)
    df['Humidity_%'] = 100 - (df['Temperature_C'] * 2).clip(0, 100)  # Simplified
    df['Drought_Duration_days'] = (df['Precipitation_mm'] < 5).astype(int).groupby(df.index // 731).cumsum()

    df['Hybrid_ID'] = np.random.choice(['P1151AM', 'DKC62-08', 'MON89034'], len(df), p=[0.4, 0.3, 0.3])
    df['WUE_g_per_mm'] = np.random.uniform(0.6, 1.0, len(df))
    df['Leaf_Water_Potential_MPa'] = np.random.uniform(-2.0, -0.5, len(df))
    df['Stomatal_Conductance_mol_m2_s'] = np.random.uniform(0.1, 0.4, len(df))
    df['Root_Depth_cm'] = np.random.uniform(80, 120, len(df))
    df['Photosynthetic_Rate_umol_m2_s'] = np.random.uniform(20, 40, len(df))
    df['Plant_Biomass_g_m2'] = np.random.uniform(100, 300, len(df))
    df['ZmDREB2A'] = np.where(df['Hybrid_ID'] == 'P1151AM', 1, np.random.choice([1, 0], len(df), p=[0.3, 0.7]))
    df['Root_QTL'] = np.random.choice([1, 0], len(df), p=[0.7, 0.3])
    df['ZmNAC'] = np.random.choice([1, 0], len(df), p=[0.5, 0.5])
    df['Planting_Density_plants_ha'] = np.random.uniform(60000, 80000, len(df))
    df['Soil_Type'] = np.random.choice(['Loam', 'Silt', 'Clay'], len(df))
    df['Growth_Stage'] = np.random.choice(['Vegetative', 'Flowering', 'Grain_Fill'], len(df))

    df['Drought_Score'] = (0.2 * df['ZmDREB2A'] + 0.2 * df['Root_QTL'] + 0.15 * df['ZmNAC'] + 
                           0.15 * df['WUE_g_per_mm'] + 0.1 * df['Soil_Moisture_%'] / 30 - 
                           0.1 * (df['Drought_Duration_days'] / 10).clip(0, 1))
    df['Drought_Score'] = np.clip(df['Drought_Score'], 0, 1)
    df['Resistance_Level'] = pd.cut(df['Drought_Score'], bins=[0, 0.6, 0.8, 1.0], 
                                    labels=['Low', 'Medium', 'High'], include_lowest=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Suggestion dataset complete! {len(df)} rows, {len(df.columns)} columns saved to {OUTPUT_FILE}")
    print("Columns:", df.columns.tolist())
    files.download(OUTPUT_FILE)


create_suggestion_dataset()