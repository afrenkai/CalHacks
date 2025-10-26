#!/usr/bin/env python3
import time
import random
import sys

def animate_listening(duration=3):
    start_time = time.time()
    frame_idx = 0

    print("\n" + "="*40)
    print("="*40)

    while time.time() - start_time < duration:
        # Clear previous frame (move cursor up)
        sys.stdout.write("\033[F" * 10)

        # Print current frame

        frame_idx += 1
        time.sleep(0.2)

    sys.stdout.write("\033[F" * 10)
    print(" " * 50)

def get_random_response():
    responses = ["yes", "no", "maybe"]
    return random.choice(responses)

def main():
    try:
        while True:
            animate_listening(duration=3)

            response = get_random_response()

            print("\n" + "="*40)
            print(f"  Response: {response.upper()}")
            print("="*40 + "\n")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped listening.")
        sys.exit(0)

if __name__ == "__main__":
    main()
