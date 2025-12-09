import numpy as np
import matplotlib.pyplot as plt


# 1. GEOMETRY (Fig. 3.2)


L = 0.2      # arm length (meters)
kf = 1.0     # thrust coefficient
km = 0.02    # yaw drag coefficient

def thrust(omega):
    """Rotor thrust from motor speed."""
    return kf * omega**2

def roll_moment(f1, f2, f3, f4):
    return L * (f2 - f4)

def pitch_moment(f1, f2, f3, f4):
    return L * (f3 - f1)

def yaw_moment(f1, f2, f3, f4):
    return km * (-f1 + f2 - f3 + f4)


# 2. CHAPTER 3 MODEL (Eq. 3.4)


# Parameters from book
T = 0.05     # motor time constant
J = 0.02     # inertia
k = 1.0      # gain
kp = 10.0    # gyro proportional gain

# State-space matrices
A = np.array([
    [-1/T, -(k*kp)/(J*T), 0],
    [1,     0,             0],
    [0,     1,             0]
])

B = np.array([[k*kp], [0], [0]])
C = np.array([[0, 0, 1]])

def attitude_dynamics(x, u):
    """One-axis model from Chapter 3."""
    return A @ x + B * u


# 3. SIMULATION


dt = 0.001
Tfinal = 3
N = int(Tfinal/dt)

x = np.zeros((3,1))   # [angular_rate, angle, integral]
u = 0.1               # small control input

history = np.zeros((N,3))
time = np.linspace(0,Tfinal,N)

for i in range(N):
    history[i] = x.flatten()
    dx = attitude_dynamics(x, u)
    x = x + dt * dx

# ===============================
# 4. PLOT
# ===============================

plt.figure(figsize=(8,4))
plt.plot(time, history[:,1], label="Angle")
plt.plot(time, history[:,0], label="Angular Rate")
plt.title("Chapter 3 Attitude Model (Eq. 3.4)")
plt.xlabel("Time (s)")
plt.grid()
plt.legend()
plt.show()


#This is exactly what the book does: they treat the drone
#  + inner P gyro loop as a single SISO plant and then design an LQG controller on top of that for nicer tracking.