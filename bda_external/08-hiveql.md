# 🐝 HiveQL

## Start Hadoop  
```bash
start-all.sh
```

## Create Hive Table  
```sql
CREATE TABLE weather_data (
    weather_date STRING,
    station_id INT,
    temp INT
);
```

## See Tables  
```sql
SHOW TABLES;
```

## Describe Table  
```sql
DESCRIBE weather_data;
```

## Load Data  
```sql
LOAD DATA LOCAL INPATH '/path' OVERWRITE INTO TABLE weather_data;
```

## Select Records  
```sql
SELECT * FROM weather_data LIMIT 5;
```

## External Table  
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

## Partitioning  
```sql
CREATE TABLE weather_p (...) 
PARTITIONED BY (year STRING) 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t';
```

```sql
ALTER TABLE weather_p ADD PARTITION (year='1990');
```

## Bucketing  
```sql
CREATE TABLE weather_b (...) 
CLUSTERED BY (station_id) INTO 4 BUCKETS 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t';
```

## Alter Table  
```sql
ALTER TABLE weather_data ADD COLUMNS (humidity INT);
```

## Drop Table  
```sql
DROP TABLE weather_data;
```

## Querying  
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

## Sorting  
```sql
SELECT * FROM weather_data ORDER BY temp DESC;
```

## Aggregation  
```sql
SELECT SUBSTR(date, 7, 4) AS year, AVG(temp) 
FROM weather_data 
GROUP BY year;
```

## Joins  
```sql
SELECT w.weather_date, w.temperature, s.location 
FROM weather_data w 
JOIN station_info s 
ON w.station_id = s.station_id;
```

## Subqueries  
```sql
SELECT * FROM weather 
WHERE temp = (
  SELECT MAX(temp) FROM weather
);
```

## Views  
```sql
CREATE VIEW temp AS 
SELECT station, AVG(temp) AS avg 
FROM weather 
GROUP BY station;

SELECT * FROM temp_summary;
```
