# ***🐷 Apache Pig (Pig Latin)***

## Start Hadoop  
```bash
start-all.sh
```

## Sample Input (`weather.txt`)  
```
1950	0   1
1950	22  1
1950	-11 1
1949	111 1
1949	78  1
1949	45  0
1951	9999    2
1951	9999    5
1952	9999    9
1953	25  0
```

## Start Pig in Local or MapReduce Mode  
```bash
pig
# or for local mode
pig -x local
```

## Load Data  
```sql
rec = LOAD 'weather.txt' AS (year:chararray, temp:int, quality:int);
```

## Describe Relation  
```sql
DESCRIBE rec;
```

> **Sample Output:**  
```
rec: {year: chararray,temp: int,quality: int}
```

## Display Records  
```sql
DUMP rec;
```

> **Sample Output:**  
```
(1950,0,1)
(1950,22,1)
(1950,-11,1)
(1949,111,1)
(1949,78,1)
(1949,45,0)
(1951,9999,2)
(1951,9999,5)
(1952,9999,9)
(1953,25,0)
```

## Filter Records  
```sql
frec = FILTER rec BY temp != 9999 AND quality IN (1);
```

## Dump Filtered Records  
```sql
DUMP frec;
```

> **Sample Output:**  
```
(1950,0,1)
(1950,22,1)
(1950,-11,1)
(1949,111,1)
(1949,78,1)
```

## Group Records by Year  
```sql
grec = GROUP rec BY year;
```

## Dump Grouped Records  
```sql
DUMP grec;
```

> **Sample Output:**  
```
(1949,{(1949,111,1),(1949,78,1),(1949,45,0)})
(1950,{(1950,0,1),(1950,22,1),(1950,-11,1)})
(1951,{(1951,9999,2),(1951,9999,5)})
(1952,{(1952,9999,9)})
(1953,{(1953,25,0)})
```

## Generate Maximum Temperature per Year  
```sql
mtemp = FOREACH grec GENERATE group AS year, MAX(rec.temp) AS mt;
```

## Display Final Result  
```sql
DUMP mtemp;
```

> **Sample Output:**  
```
(1949,111)
(1950,22)
(1951,9999)
(1952,9999)
(1953,25)
```

---