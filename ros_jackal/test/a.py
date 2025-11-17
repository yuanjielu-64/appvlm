import numpy as np
import matplotlib.pyplot as plt

# Create distance array
dist = np.linspace(0, 2.0, 200)


# Define sigmoid function with clipping
def sigmoid_velocity(dist, k, d_mid, v_min=0.5, v_max=2.0, d_max=2):
    """
    Sigmoid mapping with distance clipping
    - dist >= d_max: always return v_max
    """
    # Clip distance at d_max
    dist_clipped = np.minimum(dist, d_max)

    v = v_min + (v_max - v_min) / (1 + np.exp(-k * (dist_clipped - d_mid)))

    # For distances >= d_max, force to v_max
    if isinstance(dist, np.ndarray):
        v[dist >= d_max] = v_max
    elif dist >= d_max:
        v = v_max

    return v


# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ========== Plot 1: Your preferred configuration ==========
ax1 = axes[0, 0]
k, d_mid, d_max = 4.0, 1, 3

v = sigmoid_velocity(dist, k, d_mid, d_max=d_max)
ax1.plot(dist, v, linewidth=3, color='blue', label=f'k={k}, d_mid={d_mid}, d_max={d_max}')

# Mark the saturation point
ax1.axvline(x=d_max, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Saturation at {d_max}m')
ax1.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='v_max = 2.0 m/s')
ax1.axhline(y=1.1, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(x=0.75, color='gray', linestyle='--', alpha=0.3)

# Mark key region
ax1.fill_between([d_max, 2.0], 0, 2.2, alpha=0.2, color='green', label='Always v_max')

ax1.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
ax1.set_title('Your Preferred Configuration\nk=4, d_mid=0.75, d_max=1.5', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 2.0)
ax1.set_ylim(0, 2.2)

# ========== Plot 2: Detailed view (0-1.5m) ==========
ax2 = axes[0, 1]

dist_detailed = np.linspace(0, 1.5, 200)
v_detailed = sigmoid_velocity(dist_detailed, k, d_mid, d_max=d_max)
ax2.plot(dist_detailed, v_detailed, linewidth=3, color='blue')

# Mark key points
key_points = [0.2, 0.4, 0.75, 1.0, 1.3, 1.5]
for d in key_points:
    v_val = sigmoid_velocity(d, k, d_mid, d_max=d_max)
    ax2.scatter([d], [v_val], s=100, zorder=5, color='red')
    ax2.text(d, v_val + 0.15, f'{d}m\n{v_val:.2f}m/s',
             ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax2.axvline(x=d_mid, color='gray', linestyle='--', alpha=0.5, label=f'd_mid={d_mid}')
ax2.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
ax2.set_title('Detailed View: 0-1.5m Range\n(Active learning region)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1.5)
ax2.set_ylim(0, 2.2)

# ========== Plot 3: Compare different d_max ==========
ax3 = axes[1, 0]

for d_max_val in [1.0, 1.5, 2.0]:
    v = sigmoid_velocity(dist, k, d_mid, d_max=d_max_val)
    ax3.plot(dist, v, linewidth=2.5, label=f'd_max={d_max_val}m')
    ax3.axvline(x=d_max_val, linestyle='--', alpha=0.5)

ax3.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
ax3.set_title('Effect of d_max (Saturation Distance)\nk=4, d_mid=0.75', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 2.0)
ax3.set_ylim(0, 2.2)

# ========== Plot 4: Gradient (derivative) ==========
ax4 = axes[1, 1]

# Calculate velocity
v = sigmoid_velocity(dist, k, d_mid, d_max=d_max)
ax4_main = ax4
ax4_main.plot(dist, v, linewidth=3, color='blue', label='Velocity')
ax4_main.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
ax4_main.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold', color='blue')
ax4_main.tick_params(axis='y', labelcolor='blue')

# Calculate gradient (dv/dd)
gradient = np.gradient(v, dist)
ax4_grad = ax4_main.twinx()
ax4_grad.plot(dist, gradient, linewidth=2.5, color='red', linestyle='--', label='Gradient (dv/dd)')
ax4_grad.set_ylabel('Gradient (dv/dd)', fontsize=12, fontweight='bold', color='red')
ax4_grad.tick_params(axis='y', labelcolor='red')

ax4_main.axvline(x=d_max, color='green', linestyle='--', alpha=0.5)
ax4_main.set_title('Velocity & Gradient\n(Shows learning sensitivity)', fontsize=14, fontweight='bold')
ax4_main.legend(loc='upper left', fontsize=10)
ax4_grad.legend(loc='upper right', fontsize=10)
ax4_main.grid(True, alpha=0.3)
ax4_main.set_xlim(0, 2.0)

plt.tight_layout()

print("✅ Image saved successfully!")
print("\n📊 Configuration Summary:")
print(f"   k = {k} (steepness)")
print(f"   d_mid = {d_mid}m (midpoint)")
print(f"   d_max = {d_max}m (saturation distance)")
print(f"   v_range = [0.2, 2.0] m/s")
print("\n💡 Key Points:")
print(f"   - Distance < {d_max}m: Sigmoid mapping (learning region)")
print(f"   - Distance >= {d_max}m: Always v_max = 2.0 m/s")
print(f"   - At d={d_mid}m: v ≈ 1.1 m/s (midpoint)")

plt.show()