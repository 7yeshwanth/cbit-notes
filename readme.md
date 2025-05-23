# 🔬 **IIoT Lab Notes – External**

> ***🌟<u>All the best</u> 🚀***

---

## 🟢 ***LED***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

led = 40  # P1
g.setup(led, g.OUT)

print("Starting...")
try:
    while True:
        g.output(led, 1)
        sl(0.5)
        g.output(led, 0)
        sl(0.5)
finally:
    g.cleanup()
```

---

## 🟡 ***Multiple LEDs***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

leds = [40, 38, 36, 32]  # P1
for l in leds:
    g.setup(l, g.OUT)

cnt = 0
print("Starting...")
try:
    while True:
        g.output(leds[cnt % 4], 1)
        sl(0.5)
        g.output(leds[cnt % 4], 0)
        sl(0.5)
        cnt += 1
finally:
    g.cleanup()
```

---

## 🔵 ***Button + LED***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

led = 40  # P1
btn = 37  # P10
g.setup(led, g.OUT)
g.setup(btn, g.IN, pull_up_down=g.PUD_DOWN)  # Pull-down added

def cb():
    g.output(led, 1)
    sl(1)
    g.output(led, 0)

g.add_event_detect(btn, g.RISING, callback=cb, bouncetime=200)

print("Starting...")
try:
    while True:
        sl(1)
finally:
    g.cleanup()
```

---

## 🟣 ***Multi Button + Multi LED***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pins = {37:40, 35:38, 33:36, 31:32}  # btn:led
# P1 -> leds
# P10 -> buttons

def cb(ch):
    g.output(pins[ch], 1)
    sl(1)
    g.output(pins[ch], 0)

for led in pins.values():
    g.setup(led, g.OUT)
for btn in pins.keys():
    g.setup(btn, g.IN, pull_up_down=g.PUD_DOWN)
    g.add_event_detect(btn, g.RISING, callback=cb, bouncetime=200)

print("Starting...")
try:
    while True:
        sl(1)
finally:
    g.cleanup()
```

---

## 🔴 ***Button + Buzzer***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

buz = 19  # P18
btn = 40  # P1
g.setup(buz, g.OUT)
g.setup(btn, g.IN, pull_up_down=g.PUD_DOWN)

def cb():
    g.output(buz, 1)
    sl(1)
    g.output(buz, 0)

g.add_event_detect(btn, g.RISING, callback=cb, bouncetime=200)

print("Starting...")
try:
    while True:
        sl(1)
finally:
    g.cleanup()
```

---

## 🟠 ***PIR Motion Sensor***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pir = 40 # P1
g.setup(pir, g.IN, pull_up_down=g.PUD_DOWN)  # Pull-down added

print("Starting...")
try:
    while True:
        if g.input(pir):
            print("Motion Detected")
        else:
            print("No Motion")
        sl(1)
finally:
    g.cleanup()
```

---

## 🟤 ***PIR + Buzzer***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pir = 40 # P1
buz = 19 # P18
g.setup(pir, g.IN, pull_up_down=g.PUD_DOWN)
g.setup(buz, g.OUT)

print("Starting...")
try:
    while True:
        if g.input(pir):
            print("Motion Detected")
            g.output(buz, 1)
        else:
            print("No Motion")
            g.output(buz, 0)
        sl(1)
finally:
    g.cleanup()
```

---

## ⚪ ***DHT Sensor***

```python
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pin = 17 # P15
sensor = dht.DHT11

print('Starting...')
try:
    while True:
        h, t = dht.read_retry(sensor, pin)
        print(f'Temp: {t}, Humidity: {h}')
        sl(1)
finally:
    g.cleanup()
```

---

## 🟦 ***DHT + LCD Display***

```python
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl
from RPLCD.gpio import CharLCD

g.setmode(g.BOARD)
g.setwarnings(0)

pin = 17 # P15
rs = 33  # P11
e = 31   # P11
data_pins = [40, 38, 36, 32] # P11 -> D4567
sensor = dht.DHT11

lcd = CharLCD(cols=20, rows=4, pin_rs=rs, pin_e=e, pins_data=data_pins, numbering_mode=g.BOARD)

print('Starting...')
try:
    while True:
        h, t = dht.read_retry(sensor, pin)
        lcd.cursor_pos = (0, 0)
        lcd.write_string(f'Temp: {t} C')
        lcd.cursor_pos = (1, 0)
        lcd.write_string(f'Humid: {h}%')
        print(f'Temp: {t}, Humid: {h}')
        sl(1)
finally:
    g.cleanup()
```

---

## 🟩 ***4 Channal Relay Control***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

relay = [40, 38, 36, 32] # P1

for r in relay:
    g.setup(r, g.OUT)

cnt = 0
print("Starting...")
try:
    while True:
        g.output(relay[cnt % 4], 1)
        sl(0.5)
        g.output(relay[cnt % 4], 0)
        sl(0.5)
        cnt += 1
finally:
    g.cleanup()
```

---

## 🔴 ***Gas Sensor***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

gas = 40 # P1
g.setup(gas, g.IN, pull_up_down=g.PUD_DOWN)

print("Starting...")
try:
    while True:
        if g.input(gas):
            print("Gas Detected")
        else:
            print("No Gas")
        sl(1)
finally:
    g.cleanup()
```

---

## 🟧 ***Gas + Buzzer***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

gas = 40 # P1
buz = 19 # P18
g.setup(gas, g.IN, pull_up_down=g.PUD_DOWN)
g.setup(buz, g.OUT)

print("Starting...")
try:
    while True:
        if g.input(gas):
            print("Gas Detected")
            g.output(buz, 1)
        else:
            print("No Gas")
            g.output(buz, 0)
        sl(1)
finally:
    g.cleanup()
```

---

## 🟨 ***Servo Motor***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pwm_pin = 11 # P15
g.setup(pwm_pin, g.OUT)

servo = g.PWM(pwm_pin, 50)
servo.start(0)

dc = 2
print("Starting...")
try:
    while True:
        if dc == 13:
            dc = 2
        servo.ChangeDutyCycle(dc)
        dc += 1
        sl(0.2)
finally:
    servo.stop()
    g.cleanup()
```

---

## 🟫 ***Servo - Multiple Cycles***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pwm_pin = 11 # P15
g.setup(pwm_pin, g.OUT)

servo = g.PWM(pwm_pin, 50)
servo.start(5)

cy = int(input("Enter number of cycles: "))
print("Starting...")

try:
    for i in range(cy):
        print(f"Cycle {i+1}: Moving to 180°")
        servo.ChangeDutyCycle(12)
        sl(1)
        print(f"Cycle {i+1}: Resetting to 0°")
        servo.ChangeDutyCycle(2)
        sl(1)
finally:
    servo.stop()
    g.cleanup()
```

---

## 🌕 ***Servo - Angle Input***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pwm_pin = 11 # P15
g.setup(pwm_pin, g.OUT)

servo = g.PWM(pwm_pin, 50)
servo.start(5)

def set_angle(a):
    dt = (a // 18) + 2
    servo.ChangeDutyCycle(dt)
    sl(1)

angle = int(input("Enter angle (0-180): "))
print("Starting...")

try:
    set_angle(angle)
    print(f"Angle set to {angle} degrees")
finally:
    servo.stop()
    g.cleanup()
```

---

## 🌊 ***Soil Moisture Sensor***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

moist_pin = 7 # P15
g.setup(moist_pin, g.IN, pull_up_down=g.PUD_DOWN)

def cb(pin):
    if g.input(pin)==0:
        print("Water Detected!")
    else:
        print("No Water")

g.add_event_detect(moist_pin, g.BOTH, callback=cb, bouncetime=100)

print("Starting...")
try:
    while True:
        sl(1)
finally:
    g.cleanup()
```

---

## 🌧️ ***Rain Sensor***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

rain_pin = 7 # P15
g.setup(rain_pin, g.IN, pull_up_down=g.PUD_DOWN)

def cb(pin):
    if g.input(pin)==1:
        print("Rain Detected!")
    else:
        print("No Rain")

g.add_event_detect(rain_pin, g.BOTH, callback=cb, bouncetime=100)

print("Starting...")
try:
    while True:
        sl(1)
finally:
    g.cleanup()
```

---

## 🌐 ***ThingSpeak + DHT Sensor***

```python
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl
from requests import get

g.setmode(g.BOARD)
g.setwarnings(0)

pin = 17 # P15
sensor = dht.DHT11

print("Starting...")
try:
    while True:
        h, t = dht.read_retry(sensor, pin)
        r = get(f'https://api.thingspeak.com/update?api_key=3360pH1V21W2V1TI&field1={t}&field2={h}')
        print(f"Temp: {t}, Humidity: {h}")
        print(r.status_code)
        sl(5)
finally:
    g.cleanup()
```

---

## 📊 ***MATLAB ThingSpeak Commands***


```
>> cid=2937808
```

```
cid = 
    
    2937808

```

```
>> wk='3360RH1V21W2V1TI'
```

```
wk = 

    '3360RH1V21W2V1TI'

```

```
>> rk='5MQXM6X6YOM37GJ7'
```

```
rk = 

    '5MQXM6X6YOM37GJ7'

```

```
>> thingSpeakWrite(cid,[44,55], Fields=[1,2], WriteKey=wk)
```

```
>> thingSpeakRead(cid,Fields=[1], ReadKey=rk, NumPoints=30)
```

```
>> thingSpeakRead(cid,Fields=[1,2], ReadKey=rk, NumPoints=30)
```

```
>> a=thingSpeakRead(cid,Fields=[1], ReadKey=rk, NumPoints=30)
```

```
>> min(a)
```

```
ans = 

    31

```

```
>> max(a)
```

```
ans = 

    44

```

```
>> mean(a)
```

```
ans = 

    35.3333

```

```
>> nanmean(a)
```

```
ans = 

    35.3333

```
