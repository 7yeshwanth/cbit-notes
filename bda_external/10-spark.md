
# 🧠 Spark SQL

## Start Hadoop  
```bash
start-all.sh
```

## Start Spark SQL  
```bash
cd spark-sql/bin
./spark-sql
```

## Create Table from JSON  
```sql
CREATE TABLE flights (
    DEST_COUNTRY_NAME STRING,
    ORIGIN_COUNTRY_NAME STRING,
    count LONG
)
USING JSON
OPTIONS (path '/home/hadoop/Desktop/SparkSQLrmm/2015-summary.json');
```

## Describe Table  
```sql
DESCRIBE TABLE flights;
```

## Select All  
```sql
SELECT * FROM flights;
```

## CASE Logic  
```sql
SELECT 
    CASE 
        WHEN DEST_COUNTRY_NAME = 'United States' THEN 1
        WHEN DEST_COUNTRY_NAME = 'Egypt' THEN 0
        ELSE -1 
    END 
FROM flights;
```

## ARRAY Usage  
```sql
SELECT DEST_COUNTRY_NAME, ARRAY(1, 2, 3) FROM flights;
```

## Aggregation + Limiting  
```sql
SELECT dest_country_name 
FROM flights 
GROUP BY dest_country_name 
ORDER BY SUM(count) DESC 
LIMIT 5;
```

## Subqueries  
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

## Function Exploration  
```sql
SHOW FUNCTIONS;
SHOW SYSTEM FUNCTIONS;
SHOW USER FUNCTIONS;
SHOW FUNCTIONS "s*";
SHOW FUNCTIONS LIKE "collect*";
```
