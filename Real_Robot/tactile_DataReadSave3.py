# tactile_DataReadSave3
import socket
import struct
import time
import numpy as np
class TactileSensorClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.HOST = host
        self.PORT = port
        self.client = None

        # Format string based on Fingers struct
        self.format_string = '<qxxxx' + ('H' * 28 + 'h' * 11) * 2
        self.expected_size = struct.calcsize(self.format_string)
    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.HOST, self.PORT))
    def close(self):
        if self.client:
            self.client.close()
            self.client = None
    def read_data(self):
        if not self.client:
            raise RuntimeError("Not connected to the tactile sensor server")

        self.client.sendall(b"GET_DATA\n")
        data = b''
        while len(data) < self.expected_size:
            packet = self.client.recv(self.expected_size - len(data))
            if not packet:
                raise RuntimeError("Socket connection lost during data reception")
            data += packet

        if len(data) != self.expected_size:
            raise RuntimeError(f"Unexpected data size received: {len(data)} bytes, expected {self.expected_size}")

        tactile_values = struct.unpack(self.format_string, data)

        timestamp = tactile_values[0]
        finger0_static = tactile_values[1:29]
        finger0_dynamic_and_sensors = tactile_values[29:40]
        finger1_static = tactile_values[40:68]
        finger1_dynamic_and_sensors = tactile_values[68:79]

        D0_0 = finger0_dynamic_and_sensors[0]
        D0_1 = finger1_dynamic_and_sensors[0]
        Ax0, Ay0, Az0 = finger0_dynamic_and_sensors[1:4]
        Ax1, Ay1, Az1 = finger1_dynamic_and_sensors[1:4]
        Gx0, Gy0, Gz0 = finger0_dynamic_and_sensors[4:7]
        Gx1, Gy1, Gz1 = finger1_dynamic_and_sensors[4:7]

        data_dict = {
            'timestamp': timestamp,
            'D0_0': D0_0, 'D0_1': D0_1,
            'S_0': finger0_static,
            'S_1': finger1_static,
            'Accel_0': (Ax0, Ay0, Az0),
            'Accel_1': (Ax1, Ay1, Az1),
            'Gyro_0': (Gx0, Gy0, Gz0),
            'Gyro_1': (Gx1, Gy1, Gz1)
        }

        csv_row = [timestamp, D0_0, D0_1] + list(finger0_static) + list(finger1_static) + \
                  [Ax0, Ay0, Az0, Ax1, Ay1, Az1, Gx0, Gy0, Gz0, Gx1, Gy1, Gz1]

        return data_dict, csv_row
    def send_save_command(self):
        if not self.client:
            raise RuntimeError("Not connected")
        print("[DEBUG] Sending SAVE command")
        self.client.sendall(b"SAVE\n")

        try:
            response = self.client.recv(1024).decode('utf-8').strip()
            print(f"[DEBUG] GUI Response to SAVE: '{response}'")
        except Exception as e:
            print(f"[ERROR] Failed to read SAVE response: {e}")
    def send_stop_command(self):
        if not self.client:
            raise RuntimeError("Not connected")
        print("[DEBUG] Sending STOP command")
        self.client.sendall(b"STOP\n")

        try:
            response = self.client.recv(1024).decode('utf-8').strip()
            print(f"[DEBUG] GUI Response to STOP: '{response}'")
        except Exception as e:
            print(f"[ERROR] Failed to read STOP response: {e}")
    def send_path_command(self, path):
        if not self.client:
            raise RuntimeError("Not connected")
        command = f"PATH:{path}\n".encode('utf-8')
        print(f"[DEBUG] Sending path command: {command}")
        self.client.sendall(command)
        time.sleep(0.05)  # <-- ADD THIS SHORT DELAY!
        resp = self.client.recv(1024).decode().strip()
        print(f"[DEBUG] GUI Response to PATH: '{resp}'")
        if resp != "PATH_SET":
            print("[WARNING] GUI did not acknowledge the PATH command correctly.")
    def run(self, read=False, save=False, duration=5, save_path=None):
        print(f"[DEBUG] run(read={read}, save={save}, duration={duration}, save_path={save_path})")

        if save and save_path:
            print("[DEBUG] Setting save path...")
            self.send_path_command(save_path)
            time.sleep(0.1)

        if save:
            print("[DEBUG] Sending SAVE command...")
            self.send_save_command()
            time.sleep(0.1)

        start_time = time.time()
        print("[DEBUG] Starting main loop...")
        while time.time() - start_time < duration:
            if read:
                try:
                    data_dict, csv_row = self.read_data()
                    print("[DEBUG] Read one data row")
                    print(csv_row)
                except Exception as e:
                    print(f"[ERROR] Error reading data: {e}")
            time.sleep(0.005)

        if save:
            print("[DEBUG] Sending STOP command...")
            self.send_stop_command()
        print("[DEBUG] Run complete.")
    def run_average(self, read=False, save=False, duration=3, save_path=None):
        if save and save_path:
            self.send_path_command(save_path)
            print(">> Sent PATH command to GUI")
            time.sleep(0.1)

        if save:
            self.send_save_command()
            print(">> Sent SAVE command to GUI")
            time.sleep(0.1)

        samples = []

        start_time = time.time()
        while time.time() - start_time < duration:
            if read:
                try:
                    _, csv_row = self.read_data()
                    samples.append(csv_row)
                except Exception as e:
                    print(f"Error reading data: {e}")
            time.sleep(0.005)

        if save:
            self.send_stop_command()
            print(">> Sent STOP command to GUI")

        if not samples:
            print("No samples collected.")
            return None

        avg_sample = np.mean(samples, axis=0).tolist()
        print(">> Average sample collected:")
        print(avg_sample)

        return avg_sample

    def run_average2(self, read=False, save=False, duration=3, save_path=None, return_samples=False):
        if save and save_path:
            self.send_path_command(save_path);
            time.sleep(0.1)
        if save:
            self.send_save_command();
            time.sleep(0.1)

        samples = []
        start_time = time.time()
        while time.time() - start_time < duration:
            if read:
                try:
                    _, csv_row = self.read_data()  # <-- EXACT row written by the GUI
                    samples.append(csv_row)
                except Exception as e:
                    print(f"Error reading data: {e}")
            time.sleep(0.005)

        if save:
            self.send_stop_command()

        if not samples:
            return (None, []) if return_samples else None

        avg_sample = np.mean(samples, axis=0).tolist()
        return (avg_sample, samples) if return_samples else avg_sample

if __name__ == '__main__':
    sensor = TactileSensorClient()
    sensor.connect()

    # Example usage: save to a custom path
    # sensor.run(read=False, save=True, duration=2, save_path="/home/kourosh/fck2.csv")
    # sensor.run(read=True, save=False, duration=2, save_path="/home/kourosh/2.csv")
    # sensor.run(read=True, save=True, duration=1, save_path="/home/kourosh/2.csv")

    # # # Just read and print tactile data for 3 seconds
    average_data = sensor.run_average(read=True, save=True, duration=1, save_path="/home/kourosh/hellyeah.csv")

    sensor.close()
