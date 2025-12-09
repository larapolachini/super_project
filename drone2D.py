import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

dt = 0.1
T = 150

v = 5.0      # forward speed (m/s)
omega = 0.5  # turn rate (rad/s)

# State: [x, y, theta]
x_true = np.array([[5.0],[0.0],[np.pi/2]])
x_est  = np.zeros((3,1))
P      = np.eye(3) * 5

# Noise
Q = np.diag([0.01,0.01, np.deg2rad(1)])**2
R = np.diag([0.1,0.1])**2

H = np.array([[1,0,0],
              [0,1,0]])

def motion_model(x, u):
    theta = x[2,0]
    v, w = u[0,0], u[1,0]
    x_next = np.zeros((3,1))
    x_next[0,0] = x[0,0] + v*np.cos(theta)*dt
    x_next[1,0] = x[1,0] + v*np.sin(theta)*dt
    x_next[2,0] = x[2,0] + w*dt
    return x_next

def jacobian_F(x, u):
    theta = x[2,0]
    v = u[0,0]
    F = np.eye(3)
    F[0,2] = -v*np.sin(theta)*dt
    F[1,2] =  v*np.cos(theta)*dt
    return F

true_traj = []
est_traj  = []
meas_traj = []

for _ in range(T):
    u = np.array([[v],[omega]])

    # true motion
    x_true = motion_model(x_true, u)

    # measurement
    z = H @ x_true + np.random.multivariate_normal(np.zeros(2), R).reshape(2,1)

    # EKF prediction
    Fk = jacobian_F(x_est, u)
    x_pred = motion_model(x_est, u)
    P_pred = Fk @ P @ Fk.T + Q

    # EKF update
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(3) - K @ H) @ P_pred

    true_traj.append(x_true[:2].flatten())
    est_traj.append(x_est[:2].flatten())
    meas_traj.append(z.flatten())

true_traj = np.array(true_traj)
est_traj = np.array(est_traj)
meas = np.array(meas_traj)

plt.figure(figsize=(8,8))
plt.plot(true_traj[:,0], true_traj[:,1], 'k', label="True")
plt.plot(est_traj[:,0], est_traj[:,1], 'r', label="EKF Estimate")
plt.scatter(meas[:,0], meas[:,1], s=6, c='g', label="Measurements")
plt.axis('equal')
plt.grid()
plt.legend()
plt.title("Perfect Circular Motion EKF (Angle-Based Model)")
plt.show()
