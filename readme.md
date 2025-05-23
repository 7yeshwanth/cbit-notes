# **IIoT Lab Exteral**

> ***🌟<u>All the best</u> 🚀***

## 🟢 ***LED***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

led = 40 # p1
g.setup(led, g.OUT)

while True:
    g.output(led, 1)
    sl(0.5)
    g.output(led, 0)
    sl(0.5)
```

---

## 🟡 ***Multiple LEDs***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

leds = [40, 38, 36, 32] # p1
for l in leds:
    g.setup(l, g.OUT)

cnt = 0
while True:
    g.output(leds[cnt % 4], 1)
    sl(0.5)
    g.output(leds[cnt % 4], 0)
    sl(0.5)
    cnt += 1
```

---

## 🔵 ***Button + LED***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

led = 40 # p1
btn = 37 # p10
g.setup(led, g.OUT)
g.setup(btn, g.IN)

def callback():
    g.output(led, 1)
    sl(1)
    g.output(led, 0)

g.add_event_detect(btn, g.RISING, callback=callback, bouncetime=1)

while True:
    sl(1)
```

---

## 🟣 ***Multi Button + Multi LED***

```python
import RPi.GPIO as g
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pins = {37:40, 35:38, 33:36, 31:32}
# p1 -> leds
# p10 -> buttons

def cb(channel):
    g.output(pins[channel], 1)
    sl(1)
    g.output(pins[channel], 0)

for pin in pins.values():
    g.setup(pin, g.OUT)
for channel in pins.keys():
    g.setup(channel, g.IN)
    g.add_event_detect(channel, g.RISING, callback=cb, bouncetime=1)

while True:
    sl(1)
```

---

## ***Switch + Buzzer***

```py
```

---

## ***PIR***

```py
```

---

## ***PIR + Buzzer***

```py
```

---

## 🟠 ***DHT Sensor***

```python
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

pin = 17 # p15
sensor = dht.DHT11

print('Starting...')
while True:
    humidity, temp = dht.read_retry(sensor, pin)
    print(f'Temperature: {temp}')
    print(f'Humidity: {humidity}')
    sl(1)
```

---

## 🟤 ***DHT + LCD Display***

```python
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl
from RPLCD.gpio import CharLCD

g.setmode(g.BOARD)
g.setwarnings(0)

pin = 17 # p15
rs = 33 # p11
e = 31 # p11
data_pins = [40, 38, 36, 32] # p1 -> D4567
sensor = dht.DHT11
lcd = CharLCD(cols=20,rows=4, pin_rs=rs, pin_e=e, pins_data=data_pins, numbering_mode=g.BOARD)

print('Starting...')
while True:
    h, t = dht.read_retry(sensor, pin)
    lcd.cursor_pos = (0, 0)
    lcd.write_string(f'Temp: {t} C')
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f'Humid: {h}%')
    print(f'Temp: {t}, Humid: {h}')
    sl(1)
```

---

## ***4 Channal Relay***

```py
```

## ***Gas***

```py
```

## ***Gas + LED***

```py
```

## ***Servo***

```py
import RPi.GPIO as g
from time import sleep as sl


g.setmode(g.BOARD)
g.setwarnings(0)

op = 11 # p15

g.setup(op, g.OUT)

print('starting...')

servo = g.PWM(op, 50)

servo.start(5)

servo.ChangeDutyCycle(0)

dc = 2
while 1:
    if dc==13:
        dc=2
    servo.ChangeDutyCycle(dc)
    dc+=1
    sl(0.2)
```

## ***Servo Multi Cycle***

```py
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

## ***Servo Angle***

```py
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

## ***Soil sensor***

```py
import RPi.GPIO as g
from time import sleep as sl


g.setmode(g.BOARD)
g.setwarnings(0)

ip = 7 # p15

g.setup(ip, g.IN)

def cb(s):
    if g.input(s)==:
        print('Water Detected')
    else:
        print('No Water')

g.add_event_detect(ip, g.BOTH, callback=cb, bouncetime=100)

print('starting...')
while True:
    sl(1)
```

## ***Rain sensor***

```py
import RPi.GPIO as g
from time import sleep as sl


g.setmode(g.BOARD)
g.setwarnings(0)

ip = 7 # p15

g.setup(ip, g.IN)

def cb(s):
    if g.input(s)==1:
        print('Rain Detected')
    else:
        print('No Rain')

g.add_event_detect(ip, g.BOTH, callback=cb, bouncetime=100)

print('starting...')
while 1:
    sl(1)
```

## ***ThingSpeak + DHT sensor***

```py
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl
from requests import get

g.setmode(g.BOARD)
g.setwarnings(0)

ip = 17 # p15
sen = dht.DHT11

print('starting...')
while 1:
    h, t = dht.read_retry(sen, ip)
    r = get(f'https://api.thingspeak.com/update?api_key=3360pH1V21W2V1TI&field1={t}&field2={h}')
    print(f'Temperature:{t}')
    print(f'Humidity:{h}')
    print(r)
    sl(5)
```

## ***ThingSpeak + Matlab***

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
