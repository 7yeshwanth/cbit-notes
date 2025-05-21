
# 🧩 Hive UDF

## Start Hadoop  
```bash
start-all.sh
```
## Python File  
```python
# strip_udf.py
import sys
for l in sys.stdin:
    print(l.strip().strip('.,!?'))
```

## Make Executable  
```bash
chmod +x /path/strip_udf.py
```

## Start Hive  
```bash
hive
```

## Add File  
```sql
ADD FILE /path/strip_udf.py;
```

## List Files  
```sql
LIST FILE;
```

## Create Table  
```sql
CREATE TABLE my_table (text_column STRING);
```

## Describe Table  
```sql
DESCRIBE my_table;
```

## Load Data  
```sql
LOAD DATA LOCAL INPATH '/path' INTO TABLE my_table;
```

## Select from Table  
```sql
SELECT * FROM my_table;
```

## Run UDF  
```sql
SELECT TRANSFORM (text_column)
USING 'python3 strip_udf.py'
AS (cleaned_text)
FROM my_table;
```