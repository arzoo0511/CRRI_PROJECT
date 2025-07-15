import pickle
with open("app/saved_preds/forecasts.pkl", "rb") as f:
    results = pickle.load(f)
predictions = {name: pred for name, (pred, _) in results.items()}

print("PAVEMENT RISK REPORT")


time_interval_minutes = 5  

for i in range(len(predictions['air_temp'])):
    risks = []

    if predictions['air_temp'][i] > 0.85 and predictions['solar_radiation'][i] > 0.85:
        risks.append("🔥 Overheating – High surface temp & radiation may soften pavement")

    if 'rainfall' in predictions and predictions['rainfall'][i] > 0.6:
        risks.append("❄️ Slippery – Rainfall above safe limit, risk of skidding")

    if any(predictions.get(depth, [0])[i] > 0.9 for depth in ['s1_110', 's2_140', 's3_170']):
        risks.append("🪨 Cracking Risk – Subsurface temperatures are critically high")

    if predictions.get('s4_190', [1])[i] < 0.2:
        risks.append("⚠️ Subgrade Instability – Deep soil layer shows abnormal drop")

    if risks:
        time_minutes = i * time_interval_minutes
        hours = time_minutes // 60
        minutes = time_minutes % 60
        print(f"🚨 Time Step {i}  |  Approx Time: {hours}h {minutes}m")
        print("   Risks Detected:")
        for risk in risks:
            print(f"     • {risk}")
        print("")
