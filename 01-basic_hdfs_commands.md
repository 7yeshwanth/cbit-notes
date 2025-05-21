
# 🧾 Basic HDFS Commands  

## 1. 🔧 Start Hadoop  
```bash
start-all.sh
```

> **Sample Output:**  
```
Starting namenodes on [localhost]  
Starting datanodes  
Starting secondary namenodes [ubuntu22]  
Starting resourcemanager  
Starting nodemanagers
```

### Verify Java Processes  
```bash
jps
```

> **Sample Output:**  
```
6459 NameNode  
6598 DataNode  
6812 SecondaryNameNode  
7070 ResourceManager  
7200 NodeManager  
7709 Jps
```

---

## 2. 📁 Basic HDFS Commands  

### List Files in HDFS Root  
```bash
hadoop fs -ls /
```

> **Sample Output:**  
```
Found 12 items  
drwxr-xr-x   - hadoop supergroup          0 2024-08-14 14:23 /bad_lab_2_1  
drwxrwxr--   - bdauser bdauser1          0 2024-08-08 10:56 /bda_91  
drwxr-xr-x   - hadoop supergroup          0 2024-08-14 13:41 /bda_lab2  
-rw-r--r--   1 hadoop supergroup         28 2024-08-14 14:41 /bda_lab_2  
...
```

### Recursively List Files in Directory  
```bash
hadoop fs -ls -R /
```

> Shows full tree structure of HDFS directories and files.

---

## 3. 📝 Create a Local Text File  
```bash
gedit sample.txt
```

> **Content of `sample.txt`:**  
```
Welcome to Big Data Analytics Lab (22ADC13)...  
B.E. VI SEM AI&DS S1
```

### View Local File Content  
```bash
cat sample.txt
```

> **Sample Output:**  
```
Welcome to Big Data Analytics Lab (22ADC13)...  
B.E. VI SEM AI&DS S1
```

---

## 4. 📁 Create Directory in HDFS  
```bash
hadoop fs -mkdir /rmm_bda_lab
```

---

## 5. 📤 Upload File to HDFS  
```bash
hadoop fs -put sample.txt /rmm_bda_lab
```

---

## 6. 📥 Download File from HDFS to Local  
```bash
hadoop fs -get /rmm_bda_lab/sample.txt Desktop/rmm/sample.txt
```

---

## 7. 📄 Read File from HDFS  
```bash
hadoop fs -cat /rmm_bda_lab/sample.txt
```

> **Sample Output:**  
```
Welcome to Big Data Analytics Lab (22ADC13)...  
B.E. VI SEM AI&DS S1
```

---

## 8. 🗑️ Remove Files or Directories from HDFS  

### Delete a File  
```bash
hadoop fs -rm /rmm_bda_lab/sample.txt
```

> **Sample Output:**  
```
Deleted /rmm_bda_lab/sample.txt
```

### Delete a Directory (Recursively)  
```bash
hadoop fs -rm -r /rmm_bda_lab
```

> **Sample Output:**  
```
Deleted /rmm_bda_lab
```

---

## 9. 📂 Use `copyFromLocal` and `copyToLocal` (instead of `put`/`get`)  

### Upload Using `copyFromLocal`  
```bash
hadoop fs -copyFromLocal sample.txt /rmm_bda_lab
```

### Download Using `copyToLocal`  
```bash
hadoop fs -copyToLocal /rmm_bda_lab/sample.txt Desktop/rmm/sample1.txt
```

---

## 10. 💽 Check Disk Usage  

### Disk Usage Summary (Human-readable)  
```bash
hadoop fs -du -s -h /rmm_bda_lab
```

> **Sample Output:**  
```
68  /rmm_bda_lab
```

### Full Disk Usage of Root  
```bash
hadoop fs -df -h /
```

> **Sample Output:**  
```
Filesystem                   Size  Used  Avail  Use%  
hdfs://localhost        1024.0 G  128.0 M  1023.9 G    0%
```
