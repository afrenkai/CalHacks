import serial
import time

MY_PORT = "COM7"
BAUDRATE = 1000000

HEADER = [0xFF, 0xFF]
INST_WRITE_DATA = 0x03
ADDR_GOAL_POSITION = 0x2A

SERVOS_ZERO = [2048, 2048, 2048, 2048, 2048]
# + and - are on purpose, to see the offset gaps

class ServoController:
    def __init__(self, port="COM7", baudrate=1000000):
        self.serial = serial.Serial(port, baudrate, timeout=0.1)
        self.servo_angles = SERVOS_ZERO.copy()
        if not self.serial.is_open:
            raise serial.SerialException(f"Failed to open port {port}")

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()

    def set_zero_position(self):
        for servo_id in range(5, 0, -1):
            self.set_angle(servo_id, 0)
            time.sleep(0.3)

    def set_natural_position(self):
        self.set_angle(1, 0)
        time.sleep(0.3)
        self.set_angle(2, -40)
        time.sleep(0.3)
        self.set_angle(3, -20)
        time.sleep(0.3)
        self.set_angle(4, 60)


def set_angle(self, servo_id, angle):
        position = SERVOS_ZERO[servo_id - 1] + round(max(min(angle, 90), -90) * (512 / 45))
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        params = [ADDR_GOAL_POSITION, pos_low, pos_high]
        length = len(params) + 2  # instruction + checksum
        data = [servo_id, length, INST_WRITE_DATA] + params
        checksum = calculate_checksum(data)
        packet = bytes(HEADER + data + [checksum])
        self.serial.write(packet)

def calculate_checksum(data):
    total = sum(data) & 0xFF
    return (~total) & 0xFF

def main_example():
    ser = None
    try:
        controller = ServoController(MY_PORT, BAUDRATE)

        servo_id_str = input(f"Enter servo ID to control (1-5): ")
        if not servo_id_str.isdigit() or int(servo_id_str) not in [1, 2, 3, 4, 5]:
            print("Invalid ID. Exiting.")
            return

        servo_id = int(servo_id_str)
        print("sleeping")
        #time.sleep(2)
    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")


if __name__ == "__main__":
    main_example()