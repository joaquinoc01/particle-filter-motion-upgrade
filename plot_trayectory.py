import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load log
data = pd.read_csv("build/trajectory_log.csv")

# Compute RMSE
rmse = np.sqrt(((data["robot_x"] - data["estimate_x"])**2 + (data["robot_y"] - data["estimate_y"])**2).mean())
print(f"RMSE between ground truth and estimate: {rmse:.4f}")

# Plot trajectories
plt.plot(data["robot_x"], data["robot_y"], label="Robot (Ground Truth)", marker='o')
plt.plot(data["estimate_x"], data["estimate_y"], label="Estimated (Particle Filter)", marker='x')

# Annotate start points
plt.annotate("Start", 
             (data["robot_x"].iloc[0], data["robot_y"].iloc[0]), 
             textcoords="offset points", 
             xytext=(-10, 0), 
             ha='right', 
             va='center',
             fontsize=9, 
             color='blue')

# Optional: highlight start/end with larger markers
plt.scatter([data["robot_x"].iloc[0]], [data["robot_y"].iloc[0]], color='blue', s=50, edgecolors='black', zorder=5)
plt.scatter([data["estimate_x"].iloc[0]], [data["estimate_y"].iloc[0]], color='orange', s=50, edgecolors='black', zorder=5)

# Labels & formatting
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Robot Trajectory vs Particle Filter Estimate")
plt.legend()
plt.axis("equal")
plt.grid(True)

# Save with higher resolution
plt.savefig("build/trajectory_plot.png", dpi=300)
plt.show()
