import serial
import time
import random
import threading
import os
from pathlib import Path

# --- Constants ---
MY_PORT_WHITE = "/dev/ttyACM0"
MY_PORT_BLACK = "/dev/ttyACM1"
BAUDRATE = 1000000

HEADER = [0xFF, 0xFF]
INST_WRITE_DATA = 0x03
ADDR_GOAL_POSITION = 0x2A

SERVOS_ZERO = [2048, 2048, 2048, 2048, 2048]

# Index: 0   1    2    3   4
# Servo: 1   2    3    4   5
NATURAL_POS = [0, -40, -20, 60, 0]
FOLDING_POS = [0, -90, 90, -90, 0]

# --- TTS Integration ---
_tts_loaded = False
_tts_module = None

def load_tts():
    """Lazy load TTS module to avoid startup overhead"""
    global _tts_loaded, _tts_module
    if not _tts_loaded:
        print("Loading TTS system (this may take a moment)...")
        try:
            import tts as tts_module
            _tts_module = tts_module
            _tts_loaded = True
            print("TTS system loaded successfully.")
        except Exception as e:
            print(f"Error loading TTS: {e}")
            raise
    return _tts_module

def process_audio_file(audio_path, classifier_model_path=None):
    """
    Process an audio file and return the predicted gesture.

    Args:
        audio_path: Path to audio file
        classifier_model_path: Optional path to trained classifier weights
    Returns:
        gesture: 'yes', 'no', or 'maybe'
        text: Transcribed text
        probabilities: Class probabilities
    """
    tts = load_tts()

    print(f"\nProcessing audio: {audio_path}")
    print("Transcribing...")
    text, prediction, probabilities = tts.transcribe_and_classify(audio_path, classifier_model_path)

    print(f"Transcription: {text}")
    print(f"Classification: {prediction}")
    print(f"Probabilities: yes={probabilities[0]:.3f}, no={probabilities[1]:.3f}, maybe={probabilities[2]:.3f}")

    # Map prediction to gesture
    # The prediction from tts.py can be 'yes', 'no', or 'maybe/confusion'
    if 'yes' in prediction.lower():
        gesture = 'yes'
    elif 'no' in prediction.lower():
        gesture = 'no'
    else:
        gesture = 'maybe'

    return gesture, text, probabilities


# --- Helper Function ---
def calculate_checksum(data):
    """Calculates the checksum for the command packet."""
    total = sum(data) & 0xFF
    return (~total) & 0xFF


# --- Controller Class ---
class ServoController:
    # (No changes to this class)
    def __init__(self, port="COM7", baudrate=1000000):
        self.serial = serial.Serial(port, baudrate, timeout=0.1)
        self.servo_angles = SERVOS_ZERO.copy()
        if not self.serial.is_open:
            raise serial.SerialException(f"Failed to open port {port}")
        print(f"Successfully connected to {port}.")

    def close(self):
        if self.serial and self.serial.is_open:
            port = self.serial.port
            self.serial.close()
            print(f"Serial port {port} closed.")

    def set_zero_position(self):
        print(f"Port {self.serial.port}: Setting all servos to ZERO position...")
        for servo_id in range(5, 0, -1):
            self.set_angle(servo_id, 0)
            time.sleep(0.3)

    def set_natural_position(self):
        print(f"Port {self.serial.port}: Setting arm to NATURAL resting pose...")
        self.set_angle(5, NATURAL_POS[4]);
        time.sleep(0.2)
        self.set_angle(4, NATURAL_POS[3]);
        time.sleep(0.2)
        self.set_angle(3, NATURAL_POS[2]);
        time.sleep(0.2)
        self.set_angle(2, NATURAL_POS[1]);
        time.sleep(0.2)
        self.set_angle(1, NATURAL_POS[0]);
        time.sleep(0.2)

    def set_folded_position(self):
        print(f"Port {self.serial.port}: Setting arm to FOLDING resting pose...")
        self.set_angle(5, FOLDING_POS[4]);
        time.sleep(0.2)
        self.set_angle(4, FOLDING_POS[3]);
        time.sleep(0.2)
        self.set_angle(3, FOLDING_POS[2]);
        time.sleep(0.2)
        self.set_angle(2, FOLDING_POS[1]);
        time.sleep(0.2)
        self.set_angle(1, FOLDING_POS[0]);
        time.sleep(0.2)

    def set_angle(self, servo_id, angle):
        angle = max(min(angle, 90), -90)
        position = SERVOS_ZERO[servo_id - 1] + round(angle * (512 / 45))
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        params = [ADDR_GOAL_POSITION, pos_low, pos_high]
        length = len(params) + 2
        data = [servo_id, length, INST_WRITE_DATA] + params
        checksum = calculate_checksum(data)
        packet = bytes(HEADER + data + [checksum])
        self.serial.write(packet)


# --- Gesture Functions (Lock-Acquiring) ---

def gesture_yes(controller, lock, repetitions=3, speed=0.45):
    """ Performs a 'yes' nod. """
    with lock:
        print(f"Port {controller.serial.port}: Gesturing YES")
        neck_neutral = NATURAL_POS[3]
        neck_down = neck_neutral - 40
        for _ in range(repetitions):
            controller.set_angle(4, neck_down);
            time.sleep(speed)
            controller.set_angle(4, neck_neutral);
            time.sleep(speed * 0.8)
        controller.set_angle(4, neck_neutral);
        time.sleep(speed)
    print(f"Port {controller.serial.port}: YES complete.")


def gesture_no(controller, lock, repetitions=3, speed=0.3):
    """ Performs a 'no' head shake. """
    with lock:
        print(f"Port {controller.serial.port}: Gesturing NO")
        head_neutral = NATURAL_POS[4]
        for _ in range(repetitions):
            controller.set_angle(5, head_neutral + 60);
            time.sleep(speed)
            controller.set_angle(5, head_neutral - 60);
            time.sleep(speed)
        controller.set_angle(5, head_neutral);
        time.sleep(speed)
    print(f"Port {controller.serial.port}: NO complete.")


def gesture_maybe(controller, lock, speed=0.8):
    """ Performs a 'maybe' hunch. """
    with lock:
        print(f"Port {controller.serial.port}: Gesturing MAYBE")
        controller.set_angle(5, NATURAL_POS[4] - 40)
        controller.set_angle(4, NATURAL_POS[3] - 20)
        controller.set_angle(3, NATURAL_POS[2] + 20)
        time.sleep(speed)
        controller.set_angle(5, NATURAL_POS[4] + 40);
        time.sleep(speed)
        controller.set_natural_position();
        time.sleep(speed)
    print(f"Port {controller.serial.port}: MAYBE complete.")


# --- Idle Gestures (All must acquire lock) ---

def gesture_simple_twitch(controller, lock):
    """ 5/8 Idle Gestures: A simple, single-servo twitch. """
    with lock:
        servo_to_move = random.choice([1, 2, 3, 4, 5])
        offset = random.uniform(-15, 15) if servo_to_move not in [2, 3] else random.uniform(-10, 10)
        base_angle = NATURAL_POS[servo_to_move - 1]
        controller.set_angle(servo_to_move, base_angle + offset)
        time.sleep(random.uniform(0.5, 0.8))
        controller.set_angle(servo_to_move, base_angle)


def gesture_idle_look_around(controller, lock):
    """ 1/8 Idle: Base and head look left and right. """
    with lock:
        print(f"Port {controller.serial.port}: Idle: Look around...")
        controller.set_angle(1, NATURAL_POS[0] + 20);
        controller.set_angle(5, NATURAL_POS[4] + 40)
        time.sleep(random.uniform(1.0, 1.5))
        controller.set_angle(1, NATURAL_POS[0] - 20);
        controller.set_angle(5, NATURAL_POS[4] - 40)
        time.sleep(random.uniform(1.0, 1.5))
        controller.set_angle(1, NATURAL_POS[0]);
        controller.set_angle(5, NATURAL_POS[4])
        time.sleep(0.5)


def gesture_idle_stretch(controller, lock):
    """ 1/8 Idle: A small "stretch" up and back. """
    with lock:
        print(f"Port {controller.serial.port}: Idle: Stretch...")
        controller.set_angle(2, NATURAL_POS[1] - 20)  # Knees
        controller.set_angle(3, NATURAL_POS[2] - 20)  # Back
        controller.set_angle(4, NATURAL_POS[3] + 10)  # Neck
        time.sleep(1.2)
        controller.set_natural_position()
        time.sleep(0.5)


def gesture_idle_head_tilt(controller, lock):
    """ 1/8 Idle: A curious head tilt side-to-side. """
    with lock:
        print(f"Port {controller.serial.port}: Idle: Head tilt...")
        controller.set_angle(5, NATURAL_POS[4] + 35);
        time.sleep(random.uniform(0.8, 1.3))
        controller.set_angle(5, NATURAL_POS[4] - 35);
        time.sleep(random.uniform(0.8, 1.3))
        controller.set_angle(5, NATURAL_POS[4]);
        time.sleep(0.5)


# --- Idle Gesture Pool ---
IDLE_GESTURE_POOL = [
    gesture_simple_twitch, gesture_simple_twitch, gesture_simple_twitch,
    gesture_simple_twitch, gesture_simple_twitch,
    gesture_idle_look_around, gesture_idle_stretch, gesture_idle_head_tilt
]


# --- Idle Loop Thread ---
def idle_loop(controller, lock, stop_event, activity_event):
    """
    Runs in a background thread.
    Waits, and if no activity is running, performs a random idle gesture.
    """
    while not stop_event.is_set():
        time.sleep(random.uniform(3.0, 7.0))
        if stop_event.is_set() or activity_event.is_set():
            continue

        gesture_to_run = random.choice(IDLE_GESTURE_POOL)
        try:
            gesture_to_run(controller, lock)
        except Exception as e:
            print(f"Error in idle gesture: {e}")


# --- Coordinated Activities ---

def _dance_move_face_each_other(c_white, c_black):
    t_w = threading.Thread(target=c_white.set_angle, args=(1, -90))
    t_b = threading.Thread(target=c_black.set_angle, args=(1, 90))
    t_w.start();
    t_b.start();
    t_w.join();
    t_b.join()


def _dance_move_face_front(c_white, c_black):
    t_w = threading.Thread(target=c_white.set_angle, args=(1, NATURAL_POS[0]))
    t_b = threading.Thread(target=c_black.set_angle, args=(1, NATURAL_POS[0]))
    t_w.start();
    t_b.start();
    t_w.join();
    t_b.join()


def _dance_move_sway_mirror(c_white, c_black, speed=0.4):
    for _ in range(2):
        t_w = threading.Thread(target=c_white.set_angle, args=(1, -20))
        t_b = threading.Thread(target=c_black.set_angle, args=(1, 20))
        t_w.start();
        t_b.start();
        t_w.join();
        t_b.join()
        time.sleep(speed)
        t_w = threading.Thread(target=c_white.set_angle, args=(1, 20))
        t_b = threading.Thread(target=c_black.set_angle, args=(1, -20))
        t_w.start();
        t_b.start();
        t_w.join();
        t_b.join()
        time.sleep(speed)


def _dance_move_high_five(c_white, c_black, speed=0.6):
    _dance_move_face_each_other(c_white, c_black);
    time.sleep(0.3)
    pose = {2: 0, 3: 45, 4: 0}
    t_w = threading.Thread(target=lambda: [c_white.set_angle(s, p) for s, p in pose.items()])
    t_b = threading.Thread(target=lambda: [c_black.set_angle(s, p) for s, p in pose.items()])
    t_w.start();
    t_b.start();
    t_w.join();
    t_b.join()
    time.sleep(speed)
    t_w = threading.Thread(target=c_white.set_natural_position)
    t_b = threading.Thread(target=c_black.set_natural_position)
    t_w.start();
    t_b.start();
    t_w.join();
    t_b.join()
    time.sleep(speed)


def activity_dance(c_white, c_black, repetitions=2):
    """ Main dance routine. Assumes locks are acquired. """
    print("Activity: Let's dance!")
    for _ in range(repetitions):
        _dance_move_sway_mirror(c_white, c_black, speed=0.5)
        time.sleep(0.5)
        _dance_move_high_five(c_white, c_black)
        time.sleep(0.5)
    _dance_move_face_front(c_white, c_black)
    print("Dance finished.")


def _fall_sequence(controller, speed=0.5):
    """ Helper: The sequence of servos going limp. """
    # 1. Head (ID 5) falls
    controller.set_angle(5, FOLDING_POS[4])
    time.sleep(speed * random.uniform(0.8, 1.2))
    # 2. Neck (ID 4) falls
    controller.set_angle(4, FOLDING_POS[3])
    time.sleep(speed * random.uniform(0.8, 1.2))
    # 3. Back (ID 3) falls
    controller.set_angle(3, FOLDING_POS[2])
    time.sleep(speed * random.uniform(0.8, 1.2))
    # 4. Knee (ID 2) falls
    controller.set_angle(2, FOLDING_POS[1])
    time.sleep(speed * random.uniform(0.8, 1.2))
    # 5. Base (ID 1)
    controller.set_angle(1, FOLDING_POS[0])


def activity_lose_power(c_white, c_black):
    """ NEW: Simulates both arms losing power. Assumes locks are acquired. """
    print("Activity: Losing power...")
    # Stagger the fall
    t_w = threading.Thread(target=_fall_sequence, args=(c_white, 0.4))
    t_b = threading.Thread(target=_fall_sequence, args=(c_black, 0.5))
    t_w.start()
    time.sleep(random.uniform(0.1, 0.3))
    t_b.start()
    t_w.join();
    t_b.join()
    print("...power lost.")


def activity_wakeup(c_white, c_black):
    """ NEW: Wakes both arms up. Assumes locks are acquired. """
    print("Activity: Waking up...")
    t_w = threading.Thread(target=c_white.set_natural_position)
    t_b = threading.Thread(target=c_black.set_natural_position)
    t_w.start()
    time.sleep(random.uniform(0.1, 0.3))  # Stagger wakeup
    t_b.start()
    t_w.join();
    t_b.join()
    print("...Arms are awake.")


# --- Main Interactive Loop ---
def main():
    controller_white = None
    controller_black = None
    stop_event = threading.Event()
    activity_event = threading.Event()  # Event to pause idle
    t_white_idle = None
    t_black_idle = None

    try:
        # --- Setup ---
        controller_white = ServoController(MY_PORT_WHITE, BAUDRATE)
        controller_black = ServoController(MY_PORT_BLACK, BAUDRATE)

        white_lock = threading.Lock()
        black_lock = threading.Lock()

        print("Setting arms to natural position...")
        controller_white.set_natural_position()
        controller_black.set_natural_position()
        time.sleep(1)

        # --- Start Idle Threads ---
        stop_event.clear()
        activity_event.clear()
        t_white_idle = threading.Thread(target=idle_loop,
                                        args=(controller_white, white_lock, stop_event, activity_event))
        t_black_idle = threading.Thread(target=idle_loop,
                                        args=(controller_black, black_lock, stop_event, activity_event))
        t_white_idle.start()
        t_black_idle.start()

        print("\n--- Arms are now idling. Ready for commands. ---")
        print("Commands:")
        print("  [white/black/both]-[yes/no/maybe] - Direct gesture control")
        print("  audio <path> [target] - Process audio file (target: white/black/both, default: both)")
        print("  dance | poweroff | wakeup | exit")

        # --- Command Loop ---
        while True:
            command = input("Enter command: ").strip().lower()

            if command == 'exit':
                break

            # --- Audio Command ---
            if command.startswith('audio '):
                parts = command.split()
                if len(parts) < 2:
                    print("Usage: audio <path> [target]")
                    print("  target: white/black/both (default: both)")
                    continue

                audio_path = parts[1]
                target = parts[2] if len(parts) > 2 else 'both'

                if not os.path.exists(audio_path):
                    print(f"Error: Audio file not found: {audio_path}")
                    continue

                if target not in ['white', 'black', 'both']:
                    print(f"Invalid target: {target}. Use white/black/both")
                    continue

                if activity_event.is_set():
                    print("Cannot process audio: Arms are in 'poweroff' state. Use 'wakeup' first.")
                    continue

                try:
                    # Process audio file
                    gesture, text, probs = process_audio_file(audio_path)
                    print(f"\n>>> Detected gesture: {gesture.upper()}")
                    print(f">>> Executing on: {target.upper()}")

                    # Determine which controllers to use
                    if target == 'white':
                        controllers_locks = [(controller_white, white_lock)]
                    elif target == 'black':
                        controllers_locks = [(controller_black, black_lock)]
                    else:  # both
                        controllers_locks = [(controller_white, white_lock), (controller_black, black_lock)]

                    # Select gesture function
                    if gesture == 'yes':
                        gesture_func = gesture_yes
                    elif gesture == 'no':
                        gesture_func = gesture_no
                    else:
                        gesture_func = gesture_maybe

                    # Execute gesture
                    for i, (controller, lock) in enumerate(controllers_locks):
                        t = threading.Thread(target=gesture_func, args=(controller, lock))
                        t.start()
                        if target == 'both' and i == 0:
                            time.sleep(random.uniform(0.1, 0.4))  # Stagger 'both'

                except Exception as e:
                    print(f"Error processing audio: {e}")
                    import traceback
                    traceback.print_exc()

                continue

            # --- Dance Command ---
            if command == 'dance':
                print("\n--- Starting Activity: Dance ---")
                activity_event.set()  # Pause idle
                print("Waiting for arms to be free...")
                with white_lock:
                    with black_lock:
                        print("Arms locked. Let's dance!")
                        activity_dance(controller_white, controller_black, 2)
                        print("Returning to natural pose.")
                        controller_white.set_natural_position()
                        controller_black.set_natural_position()
                        time.sleep(1)
                activity_event.clear()  # Resume idle
                print("--- Activity finished. Resuming idle. ---")
                continue

                # --- NEW: Power Off Command ---
            elif command == 'poweroff':
                print("\n--- Starting Activity: Losing Power ---")
                activity_event.set()  # Pause idle
                print("Waiting for arms to be free...")
                with white_lock:
                    with black_lock:
                        print("Arms locked. Fading...")
                        activity_lose_power(controller_white, controller_black)
                # NOTE: We DO NOT clear the activity_event.
                # This keeps the arms "dead" and idle threads paused.
                print("--- Arms are in 'poweroff' state. Use 'wakeup' to revive. ---")
                continue

            # --- NEW: Wakeup Command ---
            elif command == 'wakeup':
                if not activity_event.is_set():
                    print("Arms are already awake.")
                    continue

                print("\n--- Starting Activity: Waking Up ---")
                with white_lock:
                    with black_lock:
                        activity_wakeup(controller_white, controller_black)
                activity_event.clear()  # Resume idle
                print("--- Activity finished. Resuming idle. ---")
                continue

            # --- Gesture Command Parsing ---
            try:
                target, gesture = command.split('-')
            except ValueError:
                print("Invalid command. Format: 'target-gesture' or activity name")
                continue

            if activity_event.is_set():
                print(
                    "Cannot perform gesture: An activity is in progress or arms are in 'poweroff' state. Use 'wakeup'.")
                continue

            # Target the correct controller and lock
            if target == 'white':
                controllers_locks = [(controller_white, white_lock)]
            elif target == 'black':
                controllers_locks = [(controller_black, black_lock)]
            elif target == 'both':
                controllers_locks = [(controller_white, white_lock), (controller_black, black_lock)]
            else:
                print(f"Unknown target: {target}");
                continue

            # Select the correct gesture function
            if gesture == 'yes':
                gesture_func = gesture_yes
            elif gesture == 'no':
                gesture_func = gesture_no
            elif gesture == 'maybe':
                gesture_func = gesture_maybe
            else:
                print(f"Unknown gesture: {gesture}");
                continue

            # --- Execute Commands in Threads ---
            for i, (controller, lock) in enumerate(controllers_locks):
                t = threading.Thread(target=gesture_func, args=(controller, lock))
                t.start()
                if target == 'both' and i == 0:
                    time.sleep(random.uniform(0.1, 0.4))  # Stagger 'both'

    except Exception as e:
        print(f"\n An unexpected error occurred in main loop: {e}")

    finally:
        print("\n--- Exiting... ---")
        stop_event.set()
        activity_event.set()
        if t_white_idle: t_white_idle.join()
        if t_black_idle: t_black_idle.join()
        print("Idle threads stopped.")

        if controller_white:
            controller_white.set_folded_position();
            controller_white.close()
        if controller_black:
            controller_black.set_folded_position();
            controller_black.close()

        print("Arms folded. Connections closed. Goodbye!")


if __name__ == "__main__":
    main()
