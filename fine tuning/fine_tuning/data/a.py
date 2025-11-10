# # train_depth_model.py
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# import joblib

# # Load your CSV
# df = pd.read_csv("output.csv")

# # Features (X) and target (y)
# X = df[["box_area", "cam_depth"]]
# y = df["true_depth"]

# # Step 1: Train (75%) + Temp (25%)
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.25, random_state=42
# )

# # Step 2: Split Temp into Validation (10%) and Test (15%)
# # Since 10/25 = 0.4, use 0.4 for validation share within the temp set
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.6, random_state=42
# )

# print(f"Data Split → Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# # Train Random Forest Regressor
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate on Validation Set
# val_preds = model.predict(X_val)
# val_mae = mean_absolute_error(y_val, val_preds)
# val_r2 = r2_score(y_val, val_preds)
# print(f"📊 Validation → MAE: {val_mae:.3f} m | R²: {val_r2:.3f}")

# # Evaluate on Test Set
# test_preds = model.predict(X_test)
# test_mae = mean_absolute_error(y_test, test_preds)
# test_r2 = r2_score(y_test, test_preds)
# print(f"🧪 Test → MAE: {test_mae:.3f} m | R²: {test_r2:.3f}")

# # Save the trained model
# joblib.dump(model, "depth_regressor.pkl")
# print("✅ Model saved as depth_regressor.pkl")

# depth_graphs_single_csv.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# LOAD YOUR SINGLE CSV
# -----------------------------
df = pd.read_csv("output.csv")  
# Must have: box_area, cam_depth, true_depth

# -----------------------------
# LOAD YOUR TRAINED MODEL
# -----------------------------
model = joblib.load("depth_regressor.pkl")

# -----------------------------
# SPLIT INTO TRAIN + TEST
# -----------------------------
X = df[["box_area", "cam_depth"]]
y = df["true_depth"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# PREDICT
# -----------------------------
train_preds = model.predict(X_train)
test_preds  = model.predict(X_test)

# MERGE BACK FOR GRAPHS
train_results = pd.DataFrame({
    "true_depth": y_train,
    "cam_depth_before": X_train["cam_depth"],
    "predicted_depth_after": train_preds
})

test_results = pd.DataFrame({
    "true_depth": y_test,
    "cam_depth_before": X_test["cam_depth"],
    "predicted_depth_after": test_preds
})

# Combine everything
all_results = pd.concat([train_results, test_results], ignore_index=True)

# -----------------------------
# METRICS
# -----------------------------
mae = mean_absolute_error(all_results["true_depth"], all_results["predicted_depth_after"])
rmse = np.sqrt(mean_squared_error(all_results["true_depth"], all_results["predicted_depth_after"]))
r2 = r2_score(all_results["true_depth"], all_results["predicted_depth_after"])

print("\n✅ MODEL PERFORMANCE:")
print(f"MAE  = {mae:.3f} meters")
print(f"RMSE = {rmse:.3f} meters")
print(f"R²   = {r2:.3f}")

# -----------------------------
# 🔵 GRAPH 1: TRUE vs PREDICTED
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(all_results["true_depth"], all_results["predicted_depth_after"], alpha=0.6)
plt.plot([all_results.true_depth.min(), all_results.true_depth.max()],
         [all_results.true_depth.min(), all_results.true_depth.max()],
         linestyle="--")
plt.xlabel("True Depth (m)")
plt.ylabel("Predicted Depth (m)")
plt.title("True vs Predicted Depth")
plt.grid()
plt.savefig("true_vs_predicted.png", dpi=300)
plt.show()

# -----------------------------
# 🟠 GRAPH 2: TRUE vs CAMERA DEPTH (BEFORE)
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(all_results["true_depth"], all_results["cam_depth_before"], color="orange", alpha=0.6)
plt.plot([all_results.true_depth.min(), all_results.true_depth.max()],
         [all_results.true_depth.min(), all_results.true_depth.max()],
         linestyle="--")
plt.xlabel("True Depth (m)")
plt.ylabel("Camera Depth (Before)")
plt.title("True vs Camera Depth (Before)")
plt.grid()
plt.savefig("true_vs_camera.png", dpi=300)
plt.show()

# -----------------------------
# 🟢 GRAPH 3: BEFORE vs AFTER (IMPROVEMENT)
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(all_results["true_depth"], all_results["cam_depth_before"], "o", label="Camera Depth (Before)", alpha=0.5)
plt.plot(all_results["true_depth"], all_results["predicted_depth_after"], "o", label="AI Predicted Depth (After)", alpha=0.5)
plt.xlabel("True Depth (m)")
plt.ylabel("Depth (m)")
plt.title("Before vs After Depth Comparison")
plt.legend()
plt.grid()
plt.savefig("before_vs_after.png", dpi=300)
plt.show()

# -----------------------------
# 🔴 GRAPH 4: ERROR DISTRIBUTION
# -----------------------------
errors = all_results["predicted_depth_after"] - al_
