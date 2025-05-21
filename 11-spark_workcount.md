
# 🔡 Spark Word Count

## Sample Text File  
```
one two three
four one two
five three four
six seven one
two four five
```

## Scala (Spark Shell)  
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

## Python (PySpark)  
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
