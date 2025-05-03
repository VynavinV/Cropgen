#include <Adafruit_NeoPixel.h>

#define PIN        6      // Data pin for the LED strip
#define NUM_LEDS   30     // Change to the number of LEDs in your strip

Adafruit_NeoPixel strip(NUM_LEDS, PIN, NEO_GRB + NEO_KHZ800);

const int analogInPin = A0; // Analog input pin

void setup() {
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'
}

void loop() {
  int sensorValue = analogRead(analogInPin); // 0-1023
  int brightness = map(sensorValue, 0, 1023, 0, 255);

  // Example: Set all LEDs to the same color, brightness controlled by analog input
  for(int i=0; i<NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(brightness, brightness, brightness)); // White
  }
  strip.show();
  delay(10);
}