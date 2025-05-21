# **IIOT Internal 2**

## **Servo Control**
```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

op = 11
g.setup(op, g.OUT)
servo = g.PWM(op, 50)

servo.start(5)
servo.ChangeDutyCycle(2)
print("Starting...")
sl(1)

servo.ChangeDutyCycle(12)
print("Turning back...")
sl(1)

servo.ChangeDutyCycle(2)
sl(1)

servo.stop()
g.cleanup()
```

## **Multi-cycle Servo Control with Cycle Number**
```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

op = 11
g.setup(op, g.OUT)
servo = g.PWM(op, 50)

servo.start(5)
cy = int(input('Enter *number of Cycles*: '))

for i in range(cy):
    print(f"Cycle {i + 1}: Starting...")
    sl(1)
    servo.ChangeDutyCycle(12)

    print(f"Cycle {i + 1}: Turning back...")
    sl(1)
    servo.ChangeDutyCycle(2)

servo.stop()
g.cleanup()
```

## **Servo Angle Control**
```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

op = 11
g.setup(op, g.OUT)
servo = g.PWM(op, 50)

servo.start(5)

def set_angle(a):
    dt = (a // 18) + 2
    servo.ChangeDutyCycle(dt)
    sl(1)

a = int(input("Enter *angle* between 0 to 180: "))
servo.ChangeDutyCycle(2)
sl(1)

set_angle(a)
print(f"Angle set to {a} degrees")

sl(1)
servo.ChangeDutyCycle(2)
sl(1)

servo.stop()
g.cleanup()
```

## **Soil Moisture Detection**
```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

ip = 7
g.setup(ip, g.IN)

def read_soil(ip):
    if g.input(ip) == 1:
        print("Soil is wet")
    else:
        print("Soil is dry")

g.add_event_detect(ip, g.BOTH, callback=read_soil, bouncetime=200)

while True:
    sl(1)
```

## **Rain Detection**
```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

ip = 7
g.setup(ip, g.IN)

def read_rain(ip):
    if g.input(ip) == 0:
        print("Rain detected")
    else:
        print("No rain detected")

g.add_event_detect(ip, g.BOTH, callback=read_rain, bouncetime=200)

while True:
    sl(1)
```
## thinkspeak explore

## thinkspeak + matlab

## **ThingSpeak + DHT (Humidity & Temperature)**
```python
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl
from urllib.request import urlopen

g.setmode(g.BOARD)
g.setwarnings(0)

ip = 7
sensor = dht.DHT11
key = ''  # Replace with your *ThingSpeak API key*

try:
    while True:
        h, t = dht.read_retry(sensor, ip)
        print(f"Humidity: {h}%")
        print(f"Temperature: {t}°C")
        url = f"http://api.thingspeak.com/update?api_key={key}&field1={h}&field2={t}"
        response = urlopen(url)
        print("Data sent to ThingSpeak")
        sl(5)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    g.cleanup()
```
