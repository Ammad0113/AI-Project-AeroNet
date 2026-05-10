

import joblib
import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Optional
from datetime import datetime


class MLPipeline:
    
    def __init__(self, model_dir: str = "data/processed/"):
        self.model_dir = model_dir
        self.demand_model = None
        self.demand_scaler = None
        self.demand_features = None
        self.anomaly_model = None
        self.anomaly_scaler = None
        self.anomaly_encoder = None
        self.anomaly_features = None
        self.is_initialized = False
    
    def load_models(self, verbose: bool = True):
        
        if verbose:
            print("\n" + "="*50)
            print("LOADING ML PIPELINE MODELS")
            print("="*50)
        
        demand_model_path = os.path.join(self.model_dir, "demand_model.pkl")
        demand_scaler_path = os.path.join(self.model_dir, "demand_scaler.pkl")
        demand_features_path = os.path.join(self.model_dir, "demand_features.pkl")
        
        if os.path.exists(demand_model_path):
            self.demand_model = joblib.load(demand_model_path)
            self.demand_scaler = joblib.load(demand_scaler_path)
            self.demand_features = joblib.load(demand_features_path)
            if verbose:
                print("Demand forecasting model loaded")
        else:
            if verbose:
                print(f"Demand model not found at {demand_model_path}")
        
        anomaly_model_path = os.path.join(self.model_dir, "anomaly_model.pkl")
        anomaly_scaler_path = os.path.join(self.model_dir, "anomaly_scaler.pkl")
        anomaly_encoder_path = os.path.join(self.model_dir, "anomaly_encoder.pkl")
        anomaly_features_path = os.path.join(self.model_dir, "anomaly_features.pkl")
        
        if os.path.exists(anomaly_model_path):
            self.anomaly_model = joblib.load(anomaly_model_path)
            self.anomaly_scaler = joblib.load(anomaly_scaler_path)
            self.anomaly_encoder = joblib.load(anomaly_encoder_path)
            self.anomaly_features = joblib.load(anomaly_features_path)
            if verbose:
                print("Anomaly detection model loaded")
        else:
            if verbose:
                print(f"Anomaly model not found at {anomaly_model_path}")
        
        self.is_initialized = True
        if verbose:
            print("="*50)
    
    def predict_demand(self, hour: int, temp: float = 20, humidity: float = 60, 
                       weather: int = 1, is_holiday: int = 0, is_weekend: int = 0) -> float:
        
        if self.demand_model is None:
            return 50 + hour * 2
        
        current_month = datetime.now().month
        current_day_of_week = datetime.now().weekday()
        is_weekend_val = 1 if current_day_of_week >= 5 else 0
        
        is_rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        
        test_features = np.array([[
            2, is_holiday, 1, weather, temp, temp, 
            humidity, 10, hour, current_month, current_day_of_week, 
            is_weekend_val, is_rush_hour
        ]])
        
        test_scaled = self.demand_scaler.transform(test_features)
        prediction = self.demand_model.predict(test_scaled)[0]
        
        return max(0, prediction)
    
    def detect_anomaly(self, battery_drop: float, speed: float, 
                       altitude_change: float, battery_pct: float, 
                       vibration: float = 0) -> Tuple[str, float]:
        
        if self.anomaly_model is None:
            if battery_drop > 15:
                return "failure", 0.70
            elif speed > 25:
                return "traj", 0.65
            else:
                return "failure", 0.50
        
        feature_cols = self.anomaly_features
        test_features = np.array([[battery_drop, speed, altitude_change, battery_pct, vibration]])
        
        test_scaled = self.anomaly_scaler.transform(test_features)
        pred = self.anomaly_model.predict(test_scaled)[0]
        prob = self.anomaly_model.predict_proba(test_scaled)[0].max()
        
        anomaly_type = self.anomaly_encoder.inverse_transform([pred])[0]
        
        return anomaly_type, prob
    
    def get_demand_forecast_for_grid(self, grid) -> Dict[Tuple[int, int], float]:
        
        forecasts = {}
        current_hour = datetime.now().hour
        
        for row in range(grid.size):
            for col in range(grid.size):
                cell = grid.get_cell(row, col)
                if cell:
                    predicted_demand = self.predict_demand(hour=current_hour)
                    forecasts[(row, col)] = predicted_demand
                    cell.demand = predicted_demand
        
        return forecasts
    
    def analyze_drone_telemetry(self, drone_id: str, battery_level: float,
                                 current_speed: float, altitude_change: float,
                                 route_deviation: float) -> Tuple[str, float, str]:
        
        battery_drop = max(0, 100 - battery_level) / 10
        
        anomaly_type, confidence = self.detect_anomaly(
            battery_drop=battery_drop,
            speed=current_speed,
            altitude_change=altitude_change,
            battery_pct=battery_level,
            vibration=route_deviation
        )
        
        if anomaly_type == "failure":
            recommendation = "Return to hub for inspection"
        elif anomaly_type == "traj":
            recommendation = "Reroute or return to hub"
        else:
            recommendation = "Continue normal operations"
        
        return anomaly_type, confidence, recommendation


def demo_ml_pipeline():
    
    print("\n" + "="*60)
    print("ML PIPELINE DEMONSTRATION")
    print("="*60)
    
    pipeline = MLPipeline(model_dir="data/processed/")
    pipeline.load_models(verbose=True)
    
    print("\n" + "="*50)
    print("DEMAND FORECASTING EXAMPLES")
    print("="*50)
    
    for hour in [8, 12, 18, 22]:
        demand = pipeline.predict_demand(hour)
        print(f"  Hour {hour:02d}:00 -> Predicted demand: {demand:.0f} units")
    
    print("\n" + "="*50)
    print("ANOMALY DETECTION EXAMPLES")
    print("="*50)
    
    test_cases = [
        ("Normal flight", 85, 12, 0.5, 0.2),
        ("Battery anomaly", 30, 8, -2.0, 0.3),
        ("Rapid descent", 70, 12, -15.0, 0.2),
        ("High speed", 75, 35, 5.0, 0.4),
        ("Critical battery", 10, 5, -5.0, 0.5)
    ]
    
    for name, battery, speed, alt_change, deviation in test_cases:
        anomaly_type, confidence, recommendation = pipeline.analyze_drone_telemetry(
            drone_id="D001",
            battery_level=battery,
            current_speed=speed,
            altitude_change=alt_change,
            route_deviation=deviation
        )
        print(f"\n  {name}:")
        print(f"    Battery: {battery}%, Speed: {speed}, Alt change: {alt_change}")
        print(f"    Detection: {anomaly_type} (confidence: {confidence:.1%})")
        print(f"    Recommendation: {recommendation}")
    
    print("\n" + "="*60)
    print("ML PIPELINE READY FOR INTEGRATION")
    print("="*60)


if __name__ == "__main__":
    demo_ml_pipeline()