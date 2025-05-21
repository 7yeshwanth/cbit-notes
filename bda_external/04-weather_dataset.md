
# 🔬 MapReduce – Weather Dataset Analysis

## 🚀 Start Hadoop  
```bash
start-all.sh
```

## 📤 Upload to HDFS  
```bash
hadoop fs -mkdir /weather_data
hadoop fs -put input.txt /weather_data
```

### Check File in HDFS  
```bash
hadoop fs -ls /weather_data
```

> **Sample Output:**  
```
-rw-r--r--   1 hadoop supergroup    12054 2025-02-01 10:43 /weather_data/input.txt
```

---

## 🗂️ Mapper Script: `mapper.py`

### Python Code  
```python
#!/usr/bin/env python3

import sys

for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) < 5:
        continue
    date_part = parts[2].split('_')[0]  # Extract date like '20060201'
    temp = parts[3]
    print(f"{date_part}\t{temp}")
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
max_temp = float('-inf')

for line in sys.stdin:
    date, temp = line.strip().split('\t')
    temp = float(temp)

    if date != current_date:
        if current_date:
            print(f"{current_date}\t{max_temp}")
        current_date = date
        max_temp = temp
    else:
        max_temp = max(max_temp, temp)

# Output the last one
if current_date:
    print(f"{current_date}\t{max_temp}")
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
-input /weather_data/input.txt \
-output /weather_data/output
```

---

## 📈 View Output  

```bash
hadoop fs -cat /weather_data/output/part-00000
```

> **Sample Output:**  
```
20060201	69.45
20060202	69.72
20060203	62.85
20060204	58.77
```

---

## 🧪 Optional: Test Locally Before Running on Cluster  

```bash
cat input.txt | ./mapper.py | sort | ./reducer.py
```

> **Sample Output:**  
```
20060201	69.45
20060202	69.72
20060203	62.85
20060204	58.77
```

