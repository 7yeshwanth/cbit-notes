# BDA Internal 2 Report

> 💡 **Note:** Start Hadoop every time  
```bash
start-all.sh
```

---

## 🐝 HiveQL

### Start Hadoop  
```bash
start-all.sh
```

### Create Hive Table  
```sql
CREATE TABLE weather_data (
    weather_date STRING,
    station_id INT,
    temp INT
);
```

### See Tables  
```sql
SHOW TABLES;
```

### Describe Table  
```sql
DESCRIBE weather_data;
```

### Load Data  
```sql
LOAD DATA LOCAL INPATH '/path' OVERWRITE INTO TABLE weather_data;
```

### Select Records  
```sql
SELECT * FROM weather_data LIMIT 5;
```

### External Table  
```sql
CREATE EXTERNAL TABLE weather_external (
    weather_date STRING,
    station_id INT,
    temp INT
)
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t' 
LOCATION '/path/data';
```

### Partitioning  
```sql
CREATE TABLE weather_p (...) 
PARTITIONED BY (year STRING) 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t';
```

```sql
ALTER TABLE weather_p ADD PARTITION (year='1990');
```

### Bucketing  
```sql
CREATE TABLE weather_b (...) 
CLUSTERED BY (station_id) INTO 4 BUCKETS 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t';
```

### Alter Table  
```sql
ALTER TABLE weather_data ADD COLUMNS (humidity INT);
```

### Drop Table  
```sql
DROP TABLE weather_data;
```

### Querying  
```sql
SELECT * FROM records;
```

```sql
SELECT year, MAX(temp) 
FROM records 
WHERE temperature != 9999 
  AND quality IN (0,1,4,5,9) 
GROUP BY year;
```

```sql
SELECT MIN(temp) FROM weather_data;
```

### Sorting  
```sql
SELECT * FROM weather_data ORDER BY temp DESC;
```

### Aggregation  
```sql
SELECT SUBSTR(date, 7, 4) AS year, AVG(temp) 
FROM weather_data 
GROUP BY year;
```

### Joins  
```sql
SELECT w.weather_date, w.temperature, s.location 
FROM weather_data w 
JOIN station_info s 
ON w.station_id = s.station_id;
```

### Subqueries  
```sql
SELECT * FROM weather 
WHERE temp = (
  SELECT MAX(temp) FROM weather
);
```

### Views  
```sql
CREATE VIEW temp AS 
SELECT station, AVG(temp) AS avg 
FROM weather 
GROUP BY station;

SELECT * FROM temp_summary;
```

---

## 🧩 Hive UDF

### Python File  
```python
# strip_udf.py
import sys
for l in sys.stdin:
    print(l.strip().strip('.,!?'))
```

### Make Executable  
```bash
chmod +x /path/strip_udf.py
```

### Start Hive  
```bash
hive
```

### Add File  
```sql
ADD FILE /path/strip_udf.py;
```

### List Files  
```sql
LIST FILE;
```

### Create Table  
```sql
CREATE TABLE my_table (text_column STRING);
```

### Describe Table  
```sql
DESCRIBE my_table;
```

### Load Data  
```sql
LOAD DATA LOCAL INPATH '/path' INTO TABLE my_table;
```

### Select from Table  
```sql
SELECT * FROM my_table;
```

### Run UDF  
```sql
SELECT TRANSFORM (text_column)
USING 'python3 strip_udf.py'
AS (cleaned_text)
FROM my_table;
```

---

## 🔡 Spark Word Count

### Sample Text File  
```
one two three
four one two
five three four
six seven one
two four five
```

### Scala (Spark Shell)  
```bash
cd spark/bin
./spark-shell
```

```scala
val dt = sparkContext.textFile("file:///home/path.txt")
val dfm = dt.flatMap(x => x.split(" ")).map(x => (x, 1))
val dr = dfm.reduceByKey((a, b) => a + b)
dr.collect()
dr.toDF("word", "count").show()
```

### Python (PySpark)  
```bash
cd spark/bin
./pyspark
```

```python
dt = sc.textFile('/path')
dfm = dt.flatMap(lambda x: x.split(' ')).map(lambda w: (w, 1))
dr = dfm.reduceByKey(lambda a, b: a + b)
dr.collect()
```

---

## 🧠 Spark SQL

### Start Spark SQL  
```bash
cd spark-sql/bin
./spark-sql
```

### Create Table from JSON  
```sql
CREATE TABLE flights (
    DEST_COUNTRY_NAME STRING,
    ORIGIN_COUNTRY_NAME STRING,
    count LONG
)
USING JSON
OPTIONS (path '/home/hadoop/Desktop/SparkSQLrmm/2015-summary.json');
```

### Describe Table  
```sql
DESCRIBE TABLE flights;
```

### Select All  
```sql
SELECT * FROM flights;
```

### CASE Logic  
```sql
SELECT 
    CASE 
        WHEN DEST_COUNTRY_NAME = 'United States' THEN 1
        WHEN DEST_COUNTRY_NAME = 'Egypt' THEN 0
        ELSE -1 
    END 
FROM flights;
```

### ARRAY Usage  
```sql
SELECT DEST_COUNTRY_NAME, ARRAY(1, 2, 3) FROM flights;
```

### Aggregation + Limiting  
```sql
SELECT dest_country_name 
FROM flights 
GROUP BY dest_country_name 
ORDER BY SUM(count) DESC 
LIMIT 5;
```

### Subqueries  
```sql
SELECT * 
FROM flights 
WHERE origin_country_name IN (
  SELECT dest_country_name 
  FROM flights 
  GROUP BY dest_country_name 
  ORDER BY SUM(count) DESC 
  LIMIT 5
);
```

```sql
SELECT * 
FROM flights f1 
WHERE EXISTS (
  SELECT 1 
  FROM flights f2 
  WHERE f1.dest_country_name = f2.origin_country_name
)
AND EXISTS (
  SELECT 1 
  FROM flights f2 
  WHERE f2.dest_country_name = f1.origin_country_name
);
```

### Function Exploration  
```sql
SHOW FUNCTIONS;
SHOW SYSTEM FUNCTIONS;
SHOW USER FUNCTIONS;
SHOW FUNCTIONS "s*";
SHOW FUNCTIONS LIKE "collect*";
```

---

## 🌊 Spark Streaming

### Streaming Code (Python)  
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

sp = SparkSession.builder.appName('t1').config('spark.ui.showConsoleProgress', 'false').getOrCreate()
sp.sparkContext.setLogLevel('ERROR')

sm = StructType([
    StructField('cid', IntegerType(), True),
    StructField('temp', DoubleType(), True)
])

df = sp.readStream.format('csv').option('header', 'true').schema(sm).load('folder')
pdf = df.groupBy('cid').agg(avg('temp').alias('avg'))
qy = pdf.writeStream.outputMode('complete').format('console').start()
qy.awaitTermination()
```

### Submit Job  
```bash
cd spark/bin
./spark-submit /path/to/streaming.py
```

### Example CSVs  
```csv
cid,temp
1,35.6
2,42.5
3,42.3
1,76.1
2,12.8
```

```csv
cid,temp
1,53.6
2,24.5
4,83.1
5,96.4
```

---

## 🔄 Kafka Setup

### Check Java  
```bash
java -version
```

### Extract Kafka  
```bash
tar -xvzf kafka_2.13-3.6.1.tgz
```

### Format Storage  
```bash
bin/kafka-storage.sh format -t $(bin/kafka-storage.sh random-uuid) -c config/kraft/server.properties
```

### Start Server  
```bash
bin/kafka-server-start.sh config/kraft/server.properties
```

### Create Topic  
```bash
bin/kafka-topics.sh --create --topic quickstart-events --bootstrap-server localhost:9092
```

### Start Producer  
```bash
bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092
```

### Start Consumer  
```bash
bin/kafka-console-consumer.sh --topic quickstart-events --from-beginning --bootstrap-server localhost:9092
```
