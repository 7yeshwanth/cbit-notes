
# 🔤 MapReduce – Word Count Application

## ✅ Start Hadoop  
```bash
start-all.sh
```

---

## 📁 Sample Input File: `input.txt`  

### Content  
```
Bus Car bus car train car bus car train bus TRAIN BUS buS caR CAR car BUS TRAIN Bus Car bus car train
```

### Upload to HDFS  
```bash
hadoop fs -mkdir -p /wordcount
hadoop fs -copyFromLocal input.txt /wordcount
```

### Check File in HDFS  
```bash
hadoop fs -ls /wordcount
```

> **Sample Output:**  
```
-rw-r--r--   1 hadoop supergroup    125 2025-01-29 11:11 /wordcount/input.txt
```

---

## 🗂️ Mapper Script: `map.py`  

### Code  
```python
#!/usr/bin/env python3

import sys

for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        print(f"{word.lower()}\t1")
```

### Make Executable  
```bash
chmod +x map.py
```

---

## 🧮 Reducer Script: `red.py`  

### Code  
```python
#!/usr/bin/env python3

current_word = None
current_count = 0
word = None

for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        continue

    if current_word == word:
        current_count += count
    else:
        if current_word:
            print(f"{current_word}\t{current_count}")
        current_word = word
        current_count = count

if current_word == word:
    print(f"{current_word}\t{current_count}")
```

### Make Executable  
```bash
chmod +x red.py
```

---

## 🚀 Run MapReduce Job  

```bash
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
-D mapreduce.job.reduces=1 \
-files map.py,red.py \
-mapper map.py \
-reducer red.py \
-input /wordcount/input.txt \
-output /wordcount/output
```

---

## 📈 View Output  

```bash
hadoop fs -cat /wordcount/output/part-00000
```

> **Sample Output:**  
```
bus	6
car	7
train	3
```

---

## 🧪 Optional: Test Locally Before Running on Cluster  

```bash
cat input.txt | ./map.py | sort -k1,1 | ./red.py
```

> **Sample Output:**  
```
bus	6
car	7
train	3
```

