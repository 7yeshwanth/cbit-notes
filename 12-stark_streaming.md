# 🌊 Spark Streaming

## Streaming Code (Python)  
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

## Submit Job  
```bash
cd spark/bin
./spark-submit /path/to/streaming.py
```

## Example CSVs  
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
