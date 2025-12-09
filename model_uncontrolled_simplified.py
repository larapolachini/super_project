#point-mass model: only translation, no rotational inerti

import numpy as np
import matplotlib.pyplot as plt

m = 1.0   # mass [kg]
g = 9.81  # gravity [m/s^2]

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

def com_dynamics(state, u):
    """
    Center-of-mass model (no attitude dynamics).
    state: [X, Y, Z, Vx, Vy, Vz]
    u:     [T, phi, theta, psi] (total thrust and commanded angles)
    """
    X, Y, Z, Vx, Vy, Vz = state
    T, phi, theta, psi = u

    # position derivatives
    pos_dot = np.array([Vx, Vy, Vz])

    # thrust in body frame (upwards)
    F_body = np.array([0.0, 0.0, -T])

    # rotate to inertial frame and add gravity
    R = rotation_matrix(phi, theta, psi)
    F_inertial = R @ F_body + np.array([0.0, 0.0, m*g])

    acc = F_inertial / m

    return np.concatenate([pos_dot, acc])

# ------------------------------
# Example: move forward via pitch
# ------------------------------
dt = 0.01
Tf = 5.0
N = int(Tf/dt)

state = np.array([0, 0, 0,   0, 0, 0], dtype=float)



# command a small pitch forward, zero roll, zero yaw
phi_cmd   = 0.0
theta_cmd = np.deg2rad(-10)  # -10 degrees nose-down
psi_cmd   = 0.0

# Adjust thrust to maintain altitude while tilted
T_hover = m * g / np.cos(theta_cmd)

states = np.zeros((N, 6))
time   = np.linspace(0, Tf, N)

for k in range(N):
    states[k] = state
    u = np.array([T_hover, phi_cmd, theta_cmd, psi_cmd])
    dx = com_dynamics(state, u)
    state = state + dt * dx

X, Y, Z, Vx, Vy, Vz = states.T

plt.figure(figsize=(8,4))
plt.plot(time, X, label="X (forward)")
plt.plot(time, Z, label="Z (down)")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Center-of-mass motion with constant 10° pitch")
plt.grid(True)
plt.legend()
plt.show()
