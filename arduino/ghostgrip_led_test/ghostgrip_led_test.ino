// Member 4 lesson 2: no external wiring needed.
// Open Serial Monitor at 115200 baud and send GRASP, RELEASE, or BREAK.
const int statusLed = LED_BUILTIN;

void setup() {
  pinMode(statusLed, OUTPUT);
  digitalWrite(statusLed, LOW);
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "GRASP" || command == "BREAK") {
      digitalWrite(statusLed, HIGH);
      Serial.println("ACK ON");
    } else if (command == "RELEASE") {
      digitalWrite(statusLed, LOW);
      Serial.println("ACK OFF");
    } else {
      Serial.println("UNKNOWN COMMAND");
    }
  }
}
