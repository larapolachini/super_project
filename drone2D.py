import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# =====================================
# Drone model (x, y, vx, vy)
# =====================================
dt = 0.1  # time step
T = 200   # number of steps

# State transition matrix
F = np.eye(4)
F[0,2] = dt
F[1,3] = dt

# Measurement matrix: we measure x, y
H = np.zeros((2,4))
H[0,0] = 1
H[1,1] = 1

# Process noise covariance
Q = 0.01 * np.eye(4)  # small process noise

# Measurement noise covariance (GPS)
R = 0.1 * np.eye(2)  # small measurement noise

# =====================================
# True state, Estimated state
# =====================================
x_true = np.zeros((4,1))
x_est  = np.zeros((4,1))

# Initial covariance
P = np.eye(4)

# Trajectory storage
true_traj = []
est_traj = []
meas_traj = []

# Circle parameters
radius = 5
omega = 0.05  # angular speed

# =====================================
# Simulation main loop
# =====================================
for k in range(T):
    # Simple circular trajectory: x = r cos(w t), y = r sin(w t)
    x = radius * np.cos(omega * k)
    y = radius * np.sin(omega * k)
    vx = -radius * omega * np.sin(omega * k)
    vy = radius * omega * np.cos(omega * k)
    
    x_true = np.array([[x], [y], [vx], [vy]])
    
    # Measurement with noise every step
    v = np.random.multivariate_normal(np.zeros(2), R).reshape(2,1)
    z = H @ x_true + v
    
    # ----- KALMAN PREDICTION -----
    x_pred = F @ x_est
    P_pred = F @ P @ F.T + Q
    
    # ----- KALMAN UPDATE -----
    y_k = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y_k
    P = (np.eye(4) - K @ H) @ P_pred
    
    # Store
    true_traj.append(x_true[:2].flatten())
    est_traj.append(x_est[:2].flatten())
    meas_traj.append(z.flatten())

true_traj = np.array(true_traj)
est_traj = np.array(est_traj)
meas = np.array(meas_traj)

# =====================================
# Plot trajectory
# =====================================
plt.figure(figsize=(8,8))
plt.plot(true_traj[:,0], true_traj[:,1], 'k', label='True')
plt.plot(est_traj[:,0], est_traj[:,1], 'r', label='KF Estimate')
plt.scatter(meas[:,0], meas[:,1], s=6, c='g', label='Measurements')
plt.title("Simple 2D Circular Trajectory with Kalman Filter")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
