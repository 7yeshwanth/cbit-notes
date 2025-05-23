# **IIoT Lab Exteral Sets**

> ***🌟<u>All the best</u> 🚀***

## ***LED***
```py
```
## ***Multiple LED***
```py
```
## ***Button + LED***
```py
```
## ***Multi Button + Mulit LED***
```py
```
## ***Switch + Buzzer***
```py
```
## ***PIR***
```py
```
## ***PIR + Buzzer***
```py
```
## ***DHT***
```py
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl

g.setmode(g.BOARD)
g.setwarnings(0)

ip = 17 # p15
sen = dht.DHT11

print('starting...')
while True:
    h, t = dht.read_retry(sen, ip)
    print(f'Temperature:{t}')
    print(f'Humidity:{h}')
    sl(1)
```
## ***DHT + LCD Display***
```py
import RPi.GPIO as g
import Adafruit_DHT as dht
from time import sleep as sl
from RPLCD.gpio import CharLCD

g.setmode(g.BOARD)
g.setwarnings(0)


ip = 17 # p15
rs=33 # p11
e=31 # p11
dtp=[40,38,36,32] # p1 -> D4567
sen = dht.DHT11
lcd = CharLCD(cols=20, rows=4,pin_rs=rs,pin_e=e,pins_data=dtp,numbering_mode=g.BOARD)

print('starting...')
while True:
    h, t = dht.read_retry(sen, ip)
    lcd.cursor_pos=(0,0)
    lcd.write_string(f'Temperature:{t}')
    lcd.cursor_pos=(1,0)
    lcd.write_string(f'Humidity:{h}')
    print(f'Temperature:{t}')
    print(f'Humidity:{h}')
    sl(1)
```
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


servo = g.PWM(op, 50)

servo.start(5)

servo.ChangeDutyCycle(0)

dc = 2
while True:
    if dc==13:
        dc=2
    servo.ChangeDutyCycle(dc)
    dc+=1
    sl(0.2)
```
## ***Servo Multi Cycle***
```py
```
## ***Servo Angle***
```py
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
while True:
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
while True:
    h, t = dht.read_retry(sen, ip)
    r = get(f'https://api.thingspeak.com/update?api_key=3360pH1V21W2V1TI&field1={t}&field2={h}')
    print(f'Temperature:{t}')
    print(f'Humidity:{h}')
    print(r)
    sl(5)
```
## ***ThingSpeak + Matlab***
```py
```