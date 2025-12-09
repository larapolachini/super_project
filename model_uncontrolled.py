import numpy as np
import matplotlib.pyplot as plt

# https://wilselby.com/research/arducopter/modeling/

# ----------------------
# Parameters (example)
# ----------------------
m = 1.0     # kg
g = 9.81    # m/s^2
L = 0.2     # arm length (m)
k_m = 0.02  # yaw torque coefficient

Jx, Jy, Jz = 0.02, 0.02, 0.04  # inertia (kg m^2)
J = np.diag([Jx, Jy, Jz])

# ----------------------
# Rotation matrix
# ----------------------
def rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R = np.array([
        [cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
        [spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
        [-sth,      cth*sphi,                   cth*cphi]
    ])
    return R

# ----------------------
# Uncontrolled quadrotor dynamics
# ----------------------
def quad_dynamics(state, thrusts):
    """
    state: 12x1 [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    thrusts: 4x1 [f1,f2,f3,f4] in Newtons
    returns: 12x1 time derivative
    """
    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
    f1, f2, f3, f4 = thrusts

    # Total thrust and torques
    T = f1 + f2 + f3 + f4
    tau_phi   = L * (f2 - f4)                  # roll
    tau_theta = L * (f3 - f1)                  # pitch
    tau_psi   = k_m * (-f1 + f2 - f3 + f4)     # yaw
    tau = np.array([tau_phi, tau_theta, tau_psi])

    # Rotation matrix
    R = rotation_matrix(phi, theta, psi)

    # Translational dynamics
    pos_dot = np.array([vx, vy, vz])
    F_body  = np.array([0.0, 0.0, -T])         # thrust in body frame (upwards)
    acc     = (R @ F_body + np.array([0.0, 0.0, m*g])) / m

    # Rotational dynamics
    omega = np.array([p, q, r])
    omega_dot = np.linalg.inv(J) @ (tau - np.cross(omega, J @ omega))

    # Euler angle rates
    tan_th = np.tan(theta)
    sec_th = 1.0 / np.cos(theta)

    phi_dot   = p + q*np.sin(phi)*tan_th + r*np.cos(phi)*tan_th
    theta_dot = q*np.cos(phi) - r*np.sin(phi)
    psi_dot   = q*np.sin(phi)*sec_th + r*np.cos(phi)*sec_th

    return np.concatenate([pos_dot, acc,
                           [phi_dot, theta_dot, psi_dot],
                           omega_dot])

# ----------------------
# Example: see it misbehave in open loop
# ----------------------
dt = 0.001
T_final = 3.0
N = int(T_final / dt)

# Start at rest, level
state = np.zeros(12)

# Thrusts: slightly more on motor 2 than 4 => roll torque
f_hover = m * g / 4
thrusts = np.array([f_hover, f_hover + 0.2, f_hover, f_hover - 0.2])

states = np.zeros((N, 12))
time = np.linspace(0, T_final, N)

for k in range(N):
    states[k] = state
    dx = quad_dynamics(state, thrusts)
    state = state + dt * dx  # Euler integration

phi = states[:, 6]
theta = states[:, 7]
psi = states[:, 8]

plt.figure()
plt.plot(time, phi, label="Roll φ")
plt.plot(time, theta, label="Pitch θ")
plt.plot(time, psi, label="Yaw ψ")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()
plt.grid(True)
plt.title("Uncontrolled quadrotor attitude (constant thrusts)")
plt.show()
