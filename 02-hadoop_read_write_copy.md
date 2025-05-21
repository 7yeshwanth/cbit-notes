

# 🧾 Working with Hadoop File System: Reading, Writing and Copying


## 1. 📖 Reading from HDFS

### a) View contents of a file  
```bash
hadoop fs -cat /rmm_bda_lab/sample.txt
```

> **Sample Output:**  
```
Welcome to Big Data Analytics Lab (22ADC13)...  
B.E. VI SEM AI&DS S1
```

### b) Display first N lines  
```bash
hadoop fs -cat /rmm_bda_lab/sample.txt | head -n 5
```

> **Sample Output:**  
```
Welcome to Big Data Analytics Lab (22ADC13)...
```

### c) Display last N lines  
```bash
hadoop fs -cat /rmm_bda_lab/sample.txt | tail -n 1
```

> **Sample Output:**  
```
B.E. VI SEM AI&DS S1
```

---

## 2. 📝 Writing to HDFS

### a) Append data to an existing file  
```bash
echo "This is new data" | hadoop fs -appendToFile - /rmm_bda_lab/sample.txt
```

> **Note:** This appends the string `"This is new data"` to `sample.txt`.

### b) Create a new file and write data  
```bash
echo "Sample Content" | hadoop fs -put - /rmm_bda_lab/sample1.txt
```

> **Verify Content:**  
```bash
hadoop fs -cat /rmm_bda_lab/sample1.txt
```

> **Sample Output:**  
```
Sample Content
```

---

## 3. 📁 Copying Files in HDFS

### a) Copy a file within HDFS  
```bash
hadoop fs -mkdir /rmm_bda_lab1
hadoop fs -cp /rmm_bda_lab/sample.txt /rmm_bda_lab1
```

> **Check Copied File:**  
```bash
hadoop fs -cat /rmm_bda_lab1/sample.txt
```

> **Sample Output:**  
```
Welcome to Big Data Analytics Lab (22ADC13)...  
B.E. VI SEM AI&DS S1  
This is new data
```

---

## 4. 🗃️ Moving Files in HDFS

### a) Move a file within HDFS  
```bash
hadoop fs -mv /rmm_bda_lab/sample.txt /rmm_bda_lab1
```

> ⚠️ **Note:** If the destination file already exists, this command will fail.  
> To avoid errors, ensure the destination path doesn't have a conflicting file before running `mv`.
