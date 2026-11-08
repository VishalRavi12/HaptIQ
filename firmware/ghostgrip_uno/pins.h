// GhostGrip pin map (Member C). Board: Adafruit Metro 328 + Grove Base Shield V2 set to 5V.
// Tested on hardware 2026-11-07: TOUCH_PIN, PIEZO_PIN and LED_GREEN_PIN.
#pragma once

const int TOUCH_PIN     = 2;   // Grove Touch Sensor v1.1 (proximity, not force)
const int BUTTON_PIN    = 3;   // Grove Button: emergency STOP, always disarms
const int LED_GREEN_PIN = 4;   // armed / heartbeat / cue flash
const int LED_RED_PIN   = 5;   // fault
const int PIEZO_PIN     = 6;   // Grove Piezo Vibration Sensor, binary output (R6 turned clockwise)
const int BUZZER_PIN    = 7;   // optional short cue
const int POT_PIN       = A0;  // Grove Rotary Angle Sensor
const int SERVO_PIN     = 9;   // TowerPro SG-5010 signal only; needs its own 4.8-6V supply

// An unplugged input pin floats and reads random values. Set to 1 once the Grove Button is on D3,
// so the emergency STOP button is active. Until then, STOP is only available over serial.
// (The #ifndef lets the Wokwi simulator turn it on without editing this file.)
#ifndef BUTTON_CONNECTED
#define BUTTON_CONNECTED 1   // Grove Button on D3, connected 2026-11-08
#endif

// Servo stays electrically idle until it has passed the hardware safety gate (plan section 8).
// Set to 1 only after the servo has been tested unloaded on a verified supply.
#ifndef SERVO_ENABLED
#define SERVO_ENABLED 0
#endif

// Safe angle window around neutral. Narrow on purpose; widen only after unloaded testing.
const int SERVO_NEUTRAL_DEG = 90;
const int SERVO_MIN_DEG     = 80;
const int SERVO_MAX_DEG     = 100;

#if SERVO_ENABLED && !BUTTON_CONNECTED
#error "The servo needs a working emergency STOP button: connect it on D3 and set BUTTON_CONNECTED to 1"
#endif
