#!/usr/bin/env python3
"""
arduino_pwm_test.py
Take user input for left and right PWM values and send to Arduino over serial.
"""
import serial
import time

# ----- Configuration -----
SERIAL_PORT = '/dev/ttyUSB0'  # Adjust as needed
BAUDRATE    = 115200


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        # Give Arduino time to reset
        time.sleep(2)
    except serial.SerialException as e:
        print(f"Failed to open serial port {SERIAL_PORT}: {e}")
        return

    print("Enter PWM values for Arduino in 'left,right' format. Type 'q' or 'exit' to quit.")
    while True:
        user_input = input("PWM> ").strip()
        if user_input.lower() in ('q', 'exit'):
            break

        # Accept comma or space separation
        parts = user_input.replace(',', ' ').split()
        if len(parts) != 2:
            print("Invalid input. Please enter two integers separated by comma or space.")
            continue
        try:
            left = int(parts[0])
            right = int(parts[1])
        except ValueError:
            print("Invalid integers. Please enter valid integer values.")
            continue

        # Compose and send command
        cmd = f'P:{left},{right}\n'
        ser.write(cmd.encode('ascii'))
        print(f"Sent: {cmd.strip()}")

    # Stop motors and close
    ser.write(b'P:0,0\n')
    ser.close()
    print("Serial connection closed. Exiting.")


if __name__ == '__main__':
    main()
