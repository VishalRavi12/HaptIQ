"""Member 4: optional serial connection to Arduino's LED-only starter firmware."""


class HardwareLink:
    def __init__(self, port=None):
        self.serial = None
        if port:
            import serial
            # Arduino Uno may reset when its serial port is opened.
            import time
            self.serial = serial.Serial(port, 115200, timeout=0.05)
            time.sleep(2)

    def send(self, event):
        if self.serial and self.serial.is_open:
            self.serial.write((event + '\n').encode('ascii'))

    def close(self):
        if self.serial:
            self.serial.close()
