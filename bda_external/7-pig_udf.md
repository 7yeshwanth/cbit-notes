
# 🧩 Pig UDFs – User Defined Functions

## Start Hadoop  
```bash
start-all.sh
```

# 📄 Source Files

## `students.txt`  
```
Alice,45
Bob,60
Charlie,80
David,30
```

## `myudf.py` – Jython UDFs for Pig  
```python
# Filter UDF
def pass_students(marks):
    if marks is None:
        return 'false'
    return 'true' if int(marks) > 50 else 'false'

# Eval UDF
def square(num):
    if num is None:
        return 0
    return int(num) * int(num)
```

## Step 1: Verify Jython Installation  
```bash
jython --version
```

> **Sample Output:**  
```
Jython 2.7.2-DEV
```

If not installed:
```bash
sudo apt install jython  # For Ubuntu/Debian
```

## Step 2: Set UDF Import Path if not set
```sql
grunt> set script.pig.udf.import.list myudf;
```

This tells Pig to look for custom functions in `myudf.py`.

## Step 3: Register Python UDF Script  
```sql
grunt> REGISTER 'myudf.py' USING jython AS myudf;
```

## Step 4: Load Student Data  
```sql
grunt> students = LOAD 'students.txt'
          USING PigStorage(',')
          AS (name:chararray, marks:int);
```

## Step 5: Apply Filter UDF  
Ensure correct return type (`'true'` or `'false'` as string):
```sql
grunt> passed_students = FILTER students BY myudf.pass_students(marks) == 'true';
grunt> DUMP passed_students;
```

> **Sample Output:**  
```
(Bob,60)
(Charlie,80)
```

## Step 6: Apply Eval UDF  
```sql
grunt> squared_values = FOREACH students GENERATE name, myudf.square(marks);
grunt> DUMP squared_values;
```

> **Sample Output:**  
```
(Alice,2025)
(Bob,3600)
(Charlie,6400)
(David,900)
```

---

---
