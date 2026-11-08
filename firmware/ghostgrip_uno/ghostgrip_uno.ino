// GhostGrip firmware v1 (Member C). Serial protocol v1, 115200 baud, newline-terminated ASCII.
//
//   Host -> board:  ARM,seq          explicit arm (refused while latched; send STOP first)
//                   KA,seq           keepalive while armed; never arms (send at least every 250 ms)
//                   CUE,seq,event,angle_deg,duration_ms
//                   STOP,seq         always disarms, even with a stale sequence number
//   Board -> host:  ACK,seq,uptime_ms,status
//                   ERR,seq,reason
//                   S,uptime_ms,ax_g,ay_g,az_g,piezo,touch,pot,armed   (20 Hz)
//
// Safety rules: boots DISARMED; button or STOP always disarms; only an explicit ARM re-arms;
// watchdog faults if the host goes quiet; malformed, out-of-range or stale commands never move anything.
// After a button press, watchdog or fault the board is latched: it needs STOP, then ARM.
// Only CONTACT cues move the servo; every other cue is lights only.
#include "pins.h"
#if SERVO_ENABLED
#include <Servo.h>
Servo servo;
#endif

enum State { DISARMED, ARMED_IDLE, CUE_ACTIVE, FAULT };
State state = DISARMED;

#ifndef WATCHDOG_MS
#define WATCHDOG_MS 500UL  // the Wokwi simulation raises this so commands can be typed by hand
#endif
const unsigned long CUE_COOLDOWN_MS = 500;
const unsigned long MAX_CUE_MS      = 500;
const unsigned long SAMPLE_EVERY_MS = 50;
const long MAX_NUMBER = 100000000L;
const int MAX_LINE = 48;
const int MAX_FIELDS = 5;
const int MAX_BAD_PACKETS = 5;

const char *CUE_EVENTS[] = {"CONTACT", "STABLE_HOLD", "SLIP", "DAMAGE"};

char line[MAX_LINE + 1];
int lineLen = 0;
bool lineInvalid = false;

long lastSeq = -1;
bool latched = false;
int badPackets = 0;
unsigned long lastHostMs = 0;
unsigned long lastCueStartMs = 0;
bool cueEverStarted = false;
unsigned long cueEndMs = 0;
unsigned long lastSampleMs = 0;
bool piezoSeen = false;

const char *stateName() {
  switch (state) {
    case DISARMED:   return "DISARMED";
    case ARMED_IDLE: return "ARMED";
    case CUE_ACTIVE: return "CUE";
    default:         return "FAULT";
  }
}

bool isArmed() {
  return state == ARMED_IDLE || state == CUE_ACTIVE;
}

void servoNeutral() {
#if SERVO_ENABLED
  if (servo.attached()) {
    servo.write(SERVO_NEUTRAL_DEG);
  }
#endif
}

void disarm(State next) {
  servoNeutral();
  digitalWrite(BUZZER_PIN, LOW);
  state = next;
}

void sendAck(long seq) {
  Serial.print("ACK,"); Serial.print(seq); Serial.print(',');
  Serial.print(millis()); Serial.print(','); Serial.println(stateName());
}

void sendErr(long seq, const char *reason) {
  Serial.print("ERR,"); Serial.print(seq); Serial.print(','); Serial.println(reason);
}

void badPacket(long seq, const char *reason) {
  sendErr(seq, reason);
  badPackets++;
  if (badPackets >= MAX_BAD_PACKETS && state != DISARMED) {
    disarm(FAULT);
    latched = true;
  }
}

bool buttonPressed() {
#if BUTTON_CONNECTED
  return digitalRead(BUTTON_PIN) == HIGH;  // active-high on the Grove Button module
#else
  return false;
#endif
}

// Parses a non-negative integer field. Returns false on anything else.
bool parseNumber(const char *text, long &out) {
  if (*text == '\0') return false;
  long value = 0;
  for (const char *p = text; *p; p++) {
    if (*p < '0' || *p > '9') return false;
    value = value * 10 + (*p - '0');
    if (value > MAX_NUMBER) return false;
  }
  out = value;
  return true;
}

bool isCueEvent(const char *name) {
  for (unsigned int i = 0; i < sizeof(CUE_EVENTS) / sizeof(CUE_EVENTS[0]); i++) {
    if (strcmp(name, CUE_EVENTS[i]) == 0) return true;
  }
  return false;
}

// Splits on commas in place. Unlike strtok it keeps empty fields, so "CUE,2,,X" is rejected.
int splitFields(char *text, char *fields[]) {
  int count = 0;
  fields[count++] = text;
  for (char *p = text; *p; p++) {
    if (*p == ',') {
      if (count == MAX_FIELDS) return -1;
      *p = '\0';
      fields[count++] = p + 1;
    }
  }
  for (int i = 0; i < count; i++) {
    if (*fields[i] == '\0') return -1;
  }
  return count;
}

void handleLine(char *text) {
  char *fields[MAX_FIELDS];
  int count = splitFields(text, fields);
  if (count < 2) { badPacket(-1, "MALFORMED"); return; }

  const char *cmd = fields[0];
  long seq;
  bool seqOk = parseNumber(fields[1], seq);

  // STOP is handled before any other check and always disarms.
  if (strcmp(cmd, "STOP") == 0 && count == 2) {
    disarm(DISARMED);
    latched = false;
    badPackets = 0;
    if (seqOk && seq > lastSeq) lastSeq = seq;
    sendAck(seqOk ? seq : -1);
    return;
  }

  if (!seqOk) { badPacket(-1, "BAD_SEQ"); return; }
  if (seq <= lastSeq) { badPacket(seq, "STALE_SEQ"); return; }

  if (strcmp(cmd, "ARM") == 0 && count == 2) {
    lastSeq = seq;
    if (buttonPressed()) { sendErr(seq, "BUTTON_HELD"); return; }
    if (latched) { sendErr(seq, "LATCHED"); return; }
    lastHostMs = millis();
    badPackets = 0;
    if (!isArmed()) state = ARMED_IDLE;
    sendAck(seq);
    return;
  }

  if (strcmp(cmd, "KA") == 0 && count == 2) {
    lastSeq = seq;
    if (!isArmed()) { sendErr(seq, "NOT_ARMED"); return; }
    lastHostMs = millis();
    sendAck(seq);
    return;
  }

  if (strcmp(cmd, "CUE") == 0 && count == 5) {
    long angle, duration;
    if (!isCueEvent(fields[2])) { badPacket(seq, "BAD_EVENT"); return; }
    if (!parseNumber(fields[3], angle) || !parseNumber(fields[4], duration)) {
      badPacket(seq, "BAD_FIELD"); return;
    }
    lastSeq = seq;
    if (state != ARMED_IDLE) {
      sendErr(seq, state == CUE_ACTIVE ? "BUSY" : "NOT_ARMED"); return;
    }
    if (angle < SERVO_MIN_DEG || angle > SERVO_MAX_DEG) { badPacket(seq, "ANGLE_RANGE"); return; }
    if (duration == 0 || duration > (long)MAX_CUE_MS) { badPacket(seq, "DURATION_RANGE"); return; }
    if (cueEverStarted && millis() - lastCueStartMs < CUE_COOLDOWN_MS) {
      sendErr(seq, "COOLDOWN"); return;
    }
    lastHostMs = millis();
    lastCueStartMs = millis();
    cueEverStarted = true;
    cueEndMs = millis() + duration;
    state = CUE_ACTIVE;
#if SERVO_ENABLED
    if (strcmp(fields[2], "CONTACT") == 0) {
      servo.write((int)angle);
    }
#endif
    badPackets = 0;
    sendAck(seq);
    return;
  }

  badPacket(seq, "UNKNOWN_CMD");
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (lineInvalid) {
        badPacket(-1, "MALFORMED");
      } else if (lineLen > 0) {
        line[lineLen] = '\0';
        handleLine(line);
      }
      lineLen = 0;
      lineInvalid = false;
    } else if (c < 32 || c > 126 || lineLen >= MAX_LINE) {
      lineInvalid = true;  // control byte, non-ASCII or too long: drop the whole line
    } else {
      line[lineLen++] = c;
    }
  }
}

void checkButton() {
  if (buttonPressed() && state != DISARMED) {
    disarm(DISARMED);
    latched = true;
  }
}

void updateTimers() {
  if (isArmed() && millis() - lastHostMs > WATCHDOG_MS) {
    disarm(FAULT);
    latched = true;
  }
  if (state == CUE_ACTIVE && (long)(millis() - cueEndMs) >= 0) {
    servoNeutral();
    state = ARMED_IDLE;
  }
}

void updateOutputs() {
  unsigned long now = millis();
  bool green;
  switch (state) {
    case DISARMED:   green = (now % 1000) < 100; break;  // short heartbeat blink
    case ARMED_IDLE: green = true; break;
    case CUE_ACTIVE: green = (now % 100) < 50; break;    // fast flicker during a cue
    default:         green = false; break;
  }
  digitalWrite(LED_GREEN_PIN, green);
  digitalWrite(LED_RED_PIN, state == FAULT);
}

void sendSample() {
  // No accelerometer connected yet, so acceleration is reported as absent, not zero.
  Serial.print("S,"); Serial.print(millis());
  Serial.print(",NA,NA,NA,");
  Serial.print(piezoSeen ? 1 : 0); Serial.print(',');
  Serial.print(digitalRead(TOUCH_PIN)); Serial.print(',');
  Serial.print(analogRead(POT_PIN)); Serial.print(',');
  Serial.println(isArmed() ? 1 : 0);
  piezoSeen = false;
}

void setup() {
  pinMode(TOUCH_PIN, INPUT);
  pinMode(BUTTON_PIN, INPUT);
  pinMode(PIEZO_PIN, INPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
#if SERVO_ENABLED
  servo.write(SERVO_NEUTRAL_DEG);  // set the target before attaching so it does not jump
  servo.attach(SERVO_PIN);
#endif
  Serial.begin(115200);
  Serial.println("READY,ghostgrip_uno,v1");
}

void loop() {
  checkButton();   // before reading commands, so a held button blocks any new cue
  readSerial();
  checkButton();
  if (digitalRead(PIEZO_PIN) == HIGH) {
    piezoSeen = true;
  }
  updateTimers();
  updateOutputs();
  if (millis() - lastSampleMs >= SAMPLE_EVERY_MS) {
    lastSampleMs = millis();
    sendSample();
  }
}
