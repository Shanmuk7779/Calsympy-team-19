"""
Heat Distribution Visualization Using Partial Derivatives
Model: T(x, y) = 100 * exp(-(x^2 + y^2) / 8)   (illustrative units)
"""

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables projection="3d")

# Step 2: Temperature function and analytical derivatives (vectorized)
def temperature(x, y):
    """T(x, y) = 100 * exp(-(x^2 + y^2) / 8)"""
    return 100.0 * np.exp(-(x**2 + y**2) / 8.0)

# Step 3: Grid on [-10, 10] x [-10, 10]; 201 points -> spacing 0.1, and 0 is a grid point
x_min, x_max = -10.0, 10.0
y_min, y_max = -10.0, 10.0
points_per_axis = 201
x_values = np.linspace(x_min, x_max, points_per_axis)
y_values = np.linspace(y_min, y_max, points_per_axis)
X, Y = np.meshgrid(x_values, y_values)      # shape (201, 201); rows = y, columns = x
dx = x_values[1] - x_values[0]
dy = y_values[1] - y_values[0]

# Step 4: Temperature over the whole grid
T = temperature(X, Y)

# Step 5: Analytical partial derivatives and gradient magnitude
dT_dx = -(X / 4.0) * T
dT_dy = -(Y / 4.0) * T
gradient_magnitude = np.sqrt(dT_dx**2 + dT_dy**2)

# Step 6: Validate with numerical derivatives.
# np.gradient on a 2D array returns derivatives along axis 0 (y, rows) first, then axis 1 (x, columns).
num_dT_dy, num_dT_dx = np.gradient(T, dy, dx)
max_err_x = np.max(np.abs(num_dT_dx - dT_dx))
max_err_y = np.max(np.abs(num_dT_dy - dT_dy))

# Step 8: Numerical results
max_temp = T.max()
row_max, col_max = np.unravel_index(np.argmax(T), T.shape)
min_temp = T.min()
row_min, col_min = np.unravel_index(np.argmin(T), T.shape)
max_grad = gradient_magnitude.max()
row_g, col_g = np.unravel_index(np.argmax(gradient_magnitude), T.shape)

print("=== Numerical results (illustrative units) ===")
print(f"Grid: {points_per_axis} x {points_per_axis} points, spacing dx = dy = {dx:.3f}")
print(f"Maximum temperature : {max_temp:.4f} at (x, y) = ({X[row_max, col_max]:.1f}, {Y[row_max, col_max]:.1f})")
print(f"Minimum on this grid: {min_temp:.3e} at (x, y) = ({X[row_min, col_min]:.1f}, {Y[row_min, col_min]:.1f})")
print("  (T only approaches 0 as distance -> infinity; this is the minimum of the finite grid only.)")
print(f"Maximum |grad T|    : {max_grad:.4f} at (x, y) = ({X[row_g, col_g]:.1f}, {Y[row_g, col_g]:.1f}), "
      f"distance from origin = {np.hypot(X[row_g, col_g], Y[row_g, col_g]):.2f}")
print(f"Max |numerical - analytical| for dT/dx: {max_err_x:.4f}")
print(f"Max |numerical - analytical| for dT/dy: {max_err_y:.4f}")
print("  (Differences come from finite spacing; np.gradient uses central differences inside, one-sided at edges.)")

# Step 7: Visualizations
extent = [x_min, x_max, y_min, y_max]

# A. 2D heat map (origin="lower" so +y points up, matching the meshgrid orientation)
fig1, ax1 = plt.subplots(figsize=(7, 6))
image = ax1.imshow(T, extent=extent, origin="lower", cmap="inferno", aspect="equal")
fig1.colorbar(image, ax=ax1, label="Temperature T(x, y)")
ax1.set_title("2D Heat Map of T(x, y) = 100·exp(-(x² + y²)/8)")
ax1.set_xlabel("x (illustrative units)")
ax1.set_ylabel("y (illustrative units)")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()

# B. Contour plot
fig2, ax2 = plt.subplots(figsize=(7, 6))
filled = ax2.contourf(X, Y, T, levels=20, cmap="inferno")
lines = ax2.contour(X, Y, T, levels=10, colors="white", linewidths=0.5)
ax2.clabel(lines, inline=True, fontsize=7, fmt="%.0f")
fig2.colorbar(filled, ax=ax2, label="Temperature T(x, y)")
ax2.plot(0, 0, "c*", markersize=12, label="Hottest point (0, 0)")
ax2.set_title("Contour Plot of Temperature Levels")
ax2.set_xlabel("x (illustrative units)")
ax2.set_ylabel("y (illustrative units)")
ax2.set_aspect("equal")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

# C. 3D surface plot
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection="3d")
surface = ax3.plot_surface(X, Y, T, cmap="inferno", edgecolor="none", rstride=2, cstride=2)
fig3.colorbar(surface, ax=ax3, shrink=0.6, pad=0.1, label="Temperature T(x, y)")
ax3.set_title("3D Surface of Temperature Distribution")
ax3.set_xlabel("x (illustrative units)")
ax3.set_ylabel("y (illustrative units)")
ax3.set_zlabel("T(x, y)")
ax3.view_init(elev=30, azim=-60)
fig3.tight_layout()

# Optional D. Gradient magnitude
fig4, ax4 = plt.subplots(figsize=(7, 6))
grad_image = ax4.imshow(gradient_magnitude, extent=extent, origin="lower", cmap="viridis", aspect="equal")
fig4.colorbar(grad_image, ax=ax4, label="|∇T| (temperature units per length unit)")
ax4.set_title("Gradient Magnitude |∇T|: Where Temperature Changes Fastest")
ax4.set_xlabel("x (illustrative units)")
ax4.set_ylabel("y (illustrative units)")
ax4.grid(True, alpha=0.3)
fig4.tight_layout()

# Step 9: Show all figures
plt.show()


# ---------------------------------------------------------------------------
# OUTPUT (printed in the console when the script is run):
#
# === Numerical results (illustrative units) ===
# Grid: 201 x 201 points, spacing dx = dy = 0.100
# Maximum temperature : 100.0000 at (x, y) = (0.0, 0.0)
# Minimum on this grid: 1.389e-09 at (x, y) = (-10.0, -10.0)
#   (T only approaches 0 as distance -> infinity; this is the minimum of the finite grid only.)
# Maximum |grad T|    : 30.3265 at (x, y) = (0.0, -2.0), distance from origin = 2.00
# Max |numerical - analytical| for dT/dx: 0.0287
# Max |numerical - analytical| for dT/dy: 0.0287
#   (Differences come from finite spacing; np.gradient uses central differences inside, one-sided at edges.)
#
# Plots shown: 2D heat map, contour plot, 3D surface, and gradient magnitude map.
# ---------------------------------------------------------------------------
