import serial
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import json
from scipy.interpolate import interp1d

# --- Utility functions to handle CSV to pressure mapping ---
def fit_and_sample(df_half, num_points=5):
    f_interp = interp1d(df_half["x"], df_half["y"], kind='cubic')
    x_new = np.linspace(df_half["x"].min(), df_half["x"].max(), num_points)
    y_new = f_interp(x_new)
    return x_new, y_new

def map_displacement_to_pressure(d):
    a, b, c = 0.143, 2.18, 0.03 - d
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    sqrt_disc = np.sqrt(discriminant)
    p1 = (-b + sqrt_disc) / (2*a)
    p2 = (-b - sqrt_disc) / (2*a)
    return p1 if p1 >= 0 else p2

def rescale(y, y_min=-0.045, y_max=0.015, out_min=1.2, out_max=15.8):
    return out_min + (y - y_min) * (out_max - out_min) / (y_max - y_min)

def csv_to_pressure_setpoints(csv_path):
    df = pd.read_csv(csv_path)
    if df.shape[0] != 14:
        raise ValueError("Expected 14 rows in CSV: 7 for each finger")

    left = df.iloc[:7]
    right = df.iloc[7:]

    x_left, y_left = fit_and_sample(left)
    x_right, y_right = fit_and_sample(right)

    y_left_scaled = rescale(y_left)
    y_right_scaled = rescale(y_right)

    pressures_left = [map_displacement_to_pressure(d) for d in y_left_scaled]
    pressures_right = [map_displacement_to_pressure(d) for d in y_right_scaled]

    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    pressures_all = pressures_left + pressures_right

    return dict(zip(labels, pressures_all))

# Define Alicat communication parameters
COM_PORTS = {
    "COM1": ["A", "B", "C", "D", "E"],
    "COM2": ["F", "G", "H", "I", "J"]
}
BAUD_RATE = 19200  # Adjust if needed

# Read pressure setpoints from CSV
CSV_PATH = "ctrlpts.csv"
full_setpoints = csv_to_pressure_setpoints(CSV_PATH)

# Group setpoints by COM port
SETPOINTS = {
    port: {addr: full_setpoints[addr] for addr in addresses}
    for port, addresses in COM_PORTS.items()
}

# Initialize serial connections
ser_connections = {}
for port in COM_PORTS.keys():
    try:
        ser_connections[port] = serial.Serial(port, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Error connecting to {port}: {e}")
        exit(1)

# Data storage for real-time plotting
pressure_data = {port: [] for port in COM_PORTS.keys()}
time_data = []

# Function to send pressure setpoint command
def set_pressure(port, address, pressure_value):
    cmd = f"{address}SP{pressure_value:.2f}\r"
    ser_connections[port].write(cmd.encode())

# Function to read pressure values
def read_pressure(port, address):
    cmd = f"{address}\r"
    ser_connections[port].write(cmd.encode())
    time.sleep(0.1)
    response = ser_connections[port].readline().decode().strip()
    return response

# Thread function to continuously update pressure readings
def update_pressure_data():
    global time_data
    while True:
        timestamp = time.time()
        time_data.append(timestamp)
        for port, addresses in COM_PORTS.items():
            total_pressure = 0
            valid_readings = 0
            for addr in addresses:
                response = read_pressure(port, addr)
                try:
                    pressure_value = float(response.split()[1])
                    total_pressure += pressure_value
                    valid_readings += 1
                except (IndexError, ValueError):
                    print(f"Error parsing response from {addr} on {port}: {response}")
                    continue
            if valid_readings > 0:
                avg_pressure = total_pressure / valid_readings
                pressure_data[port].append(avg_pressure)
            else:
                pressure_data[port].append(None)
        time.sleep(0.5)

# Function to initialize pressure control
def initialize_pressure_control():
    for port, addresses in COM_PORTS.items():
        for addr in addresses:
            set_pressure(port, addr, SETPOINTS[port][addr])
            time.sleep(0.1)

# Real-time plotting function
def animate(i):
    plt.cla()
    if time_data:
        for port in COM_PORTS.keys():
            if pressure_data[port]:
                plt.plot(time_data[-50:], pressure_data[port][-50:], label=f"{port} Pressure")
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (psi)")
        plt.legend()
        plt.title("Real-Time Alicat Pressure Monitoring")

# Start pressure data collection thread
pressure_thread = threading.Thread(target=update_pressure_data, daemon=True)
pressure_thread.start()

# Initialize pressure control
initialize_pressure_control()

# Set up real-time plot
fig = plt.figure()
ani = animation.FuncAnimation(fig, animate, interval=500)
plt.show()

# Close serial connections when done
for ser in ser_connections.values():
    ser.close()
