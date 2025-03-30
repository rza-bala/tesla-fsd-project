import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
input_csv = ROOT_DIR / "relevant_signals_obd.csv"
output_dir = ROOT_DIR / "organized_signals"
output_dir.mkdir(exist_ok=True)

# Load data
df_signals = pd.read_csv(input_csv)

# Defined categories and clearly matched signals
categories = {
    "Vehicle_Dynamics": ["SteeringAngle129","SteeringSpeed129","DI_vehicleSpeed","DI_uiSpeed","DI_accelPedalPos","DI_gear"],
    "Torque_Powertrain": ["DIR_torqueActual","DIR_torqueCommand","DIR_axleSpeed","RearTorqueRequest1D8","RearTorque1D8"],
    "Positioning_Navigation": ["Latitude","Longitude","Altitude","AltitudeAccuracy","Speed","SpeedAccuracy","DistanceTrip","Epoch","GPSLatitude04F","GPSLongitude04F","GPSAccuracy04F"],
    "IMU_Orientation": ["AngularRateX","AngularRateY","AngularRateZ","AccelerationX","AccelerationY","AccelerationZ"],
    "System_Status_Vehicle_Info": ["UnixTimeSeconds528","UTCseconds318","UTCminutes318","Odometer3B6","GearLeverPosition229"],
    "Environmental_Solar": ["VCRIGHT_estimatedThsSolarLoad","VCRIGHT_thsSolarLoadInfrared","VCRIGHT_thsSolarLoadVisible","VCRIGHT_thsHumidity","UI_solarAzimuthAngleCarRef","UI_solarElevationAngle","UI_solarAzimuthAngle"],
    "Interior_Controls": ["CP_doorPot","SCCM_turnIndicatorStalkStatus","DAS_turnIndicatorRequest"],
    "ADAS_Map_Road": ["UI_nextBranchDist","UI_streetCount","UI_mapSpeedLimit","UI_roadClass"],
    "Elevation_Terrain": ["Elevation3D8"],
}

# Separate and export clearly structured CSV files
for category, signals in categories.items():
    df_category = df_signals[df_signals["Signal"].isin(signals)]
    output_csv = output_dir / f"{category.lower()}_signals.csv"
    df_category.to_csv(output_csv, index=False)
    print(f"✅ Saved '{category}' signals to: {output_csv}")
