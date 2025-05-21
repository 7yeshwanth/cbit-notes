# 🚖 Uber Data Analysis using MapReduce 

## 🚀 Start Hadoop  
```bash
start-all.sh
```
---


## 📤 Upload to HDFS  
```bash
hadoop fs -mkdir /uber_data
hadoop fs -put input.csv /uber_data
```

### Check File in HDFS  
```bash
hadoop fs -ls /uber_data
```

> **Sample Output:**  
```
-rw-r--r--   1 hadoop supergroup    862 2025-02-01 12:21 /uber_data/input.csv
```

---

## 🗂️ Mapper Script: `mapper.py`

### Python Code  
```python
#!/usr/bin/env python3

import sys
import csv

for row in csv.reader(sys.stdin):
    if row[0].lower() == 'date/time':
        continue  # skip header
    date_time = row[0]
    date = date_time.split()[0]  # extract date only
    print(f"{date}\t1")
```

### Make Executable  
```bash
chmod +x mapper.py
```

---

## 🧮 Reducer Script: `reducer.py`

### Python Code  
```python
#!/usr/bin/env python3

import sys

current_date = None
total_count = 0

for line in sys.stdin:
    line = line.strip()
    date, count = line.split('\t')

    try:
        count = int(count)
    except ValueError:
        continue

    if current_date == date:
        total_count += count
    else:
        if current_date:
            print(f"{current_date}\t{total_count}")
        current_date = date
        total_count = count

# Print last result
if current_date == date:
    print(f"{current_date}\t{total_count}")
```

### Make Executable  
```bash
chmod +x reducer.py
```

---

## 🚀 Run MapReduce Job  

```bash
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
-D mapreduce.job.reduces=1 \
-files mapper.py,reducer.py \
-mapper mapper.py \
-reducer reducer.py \
-input /uber_data/input.csv \
-output /uber_data/output
```

---

## 📈 View Output  

```bash
hadoop fs -cat /uber_data/output/part-00000
```

> **Sample Output:**  
```
04-01-2014	10
04-02-2014	10
```

---

## 🧪 Optional: Test Locally Before Running on Cluster  

```bash
cat input.csv | ./mapper.py | sort | ./reducer.py
```

> **Sample Output:**  
```
04-01-2014	10
04-02-2014	10
```

