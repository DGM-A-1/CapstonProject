// Arduino PWM Control Sketch
// Listens on Serial for commands of the form "P:leftPWM,rightPWM\n"
// and drives two DC motors via an L298N (or similar) driver.
// Board: ESP8266 Wemos D1 (D# pins map to GPIO# as shown below)

//////////////////////////////////////////////////
// Motor driver pin definitions
//////////////////////////////////////////////////

// RIGHT motor (M1)
const int RightMotor_E_pin = 14;   // ENA (PWM) → D5 (GPIO14)
const int RightMotor_1_pin = 4;    // IN1       → D2 (GPIO4)
const int RightMotor_2_pin = 2;    // IN2       → D4 (GPIO2)

// LEFT motor (M2)
const int LeftMotor_3_pin  = 12;   // IN3       → D6 (GPIO12)
const int LeftMotor_4_pin  = 13;   // IN4       → D7 (GPIO13)
const int LeftMotor_E_pin  = 15;   // ENB (PWM) → D8 (GPIO15)

//////////////////////////////////////////////////
// Serial input buffering
//////////////////////////////////////////////////

String inputString = "";
bool stringComplete = false;

//////////////////////////////////////////////////
// Setup
//////////////////////////////////////////////////
void setup() {
  Serial.begin(115200);
  inputString.reserve(64);

  // Configure motor pins
  pinMode(RightMotor_E_pin, OUTPUT);
  pinMode(RightMotor_1_pin, OUTPUT);
  pinMode(RightMotor_2_pin, OUTPUT);
  pinMode(LeftMotor_3_pin, OUTPUT);
  pinMode(LeftMotor_4_pin, OUTPUT);
  pinMode(LeftMotor_E_pin, OUTPUT);

  // Ensure motors are stopped at startup
  stopMotors();
}

//////////////////////////////////////////////////
// Main loop
//////////////////////////////////////////////////
void loop() {
  // Read incoming serial data
  serialEvent();

  // When a full command line is received, process it
  if (stringComplete) {
    processPWMCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}

//////////////////////////////////////////////////
// Serial event handler
// Reads characters until it sees '\n'
//////////////////////////////////////////////////
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    }
    else if (inChar != '\r') {
      inputString += inChar;
    }
  }
}

//////////////////////////////////////////////////
// Command parser
// Expected format: P:leftPWM,rightPWM
//////////////////////////////////////////////////
void processPWMCommand(const String &cmd) {
  if (!cmd.startsWith("P:")) return;

  int commaIndex = cmd.indexOf(',', 2);
  if (commaIndex < 0) return;

  // Extract numeric substrings
  String leftStr  = cmd.substring(2, commaIndex);
  String rightStr = cmd.substring(commaIndex + 1);

  int leftPWM  = leftStr.toInt();
  int rightPWM = rightStr.toInt();

  // Apply to motors
  setLeftMotor(leftPWM);
  setRightMotor(rightPWM);
}

//////////////////////////////////////////////////
// Motor control functions
//////////////////////////////////////////////////

// Left motor: pwm > 0 → forward, < 0 → reverse, = 0 → stop
void setLeftMotor(int pwm) {
  if (pwm > 0) {
    digitalWrite(LeftMotor_3_pin, HIGH);
    digitalWrite(LeftMotor_4_pin, LOW);
    analogWrite(LeftMotor_E_pin, pwm);
  }
  else if (pwm < 0) {
    digitalWrite(LeftMotor_3_pin, LOW);
    digitalWrite(LeftMotor_4_pin, HIGH);
    analogWrite(LeftMotor_E_pin, -pwm);
  }
  else {
    digitalWrite(LeftMotor_3_pin, LOW);
    digitalWrite(LeftMotor_4_pin, LOW);
    analogWrite(LeftMotor_E_pin, 0);
  }
}

// Right motor: pwm > 0 → forward, < 0 → reverse, = 0 → stop
void setRightMotor(int pwm) {
  if (pwm > 0) {
    digitalWrite(RightMotor_1_pin, HIGH);
    digitalWrite(RightMotor_2_pin, LOW);
    analogWrite(RightMotor_E_pin, pwm);
  }
  else if (pwm < 0) {
    digitalWrite(RightMotor_1_pin, LOW);
    digitalWrite(RightMotor_2_pin, HIGH);
    analogWrite(RightMotor_E_pin, -pwm);
  }
  else {
    digitalWrite(RightMotor_1_pin, LOW);
    digitalWrite(RightMotor_2_pin, LOW);
    analogWrite(RightMotor_E_pin, 0);
  }
}

//////////////////////////////////////////////////
// Emergency stop / helper
//////////////////////////////////////////////////
void stopMotors() {
  analogWrite(LeftMotor_E_pin, 0);
  analogWrite(RightMotor_E_pin, 0);
  digitalWrite(LeftMotor_3_pin, LOW);
  digitalWrite(LeftMotor_4_pin, LOW);
  digitalWrite(RightMotor_1_pin, LOW);
  digitalWrite(RightMotor_2_pin, LOW);
}
