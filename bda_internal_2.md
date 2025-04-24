# BDA Internal 2

> Note: start hadoop everytime
    ```
    start-all.sh
    ```

## HiveQL
- start hadoop
    ```
    start-all.sh
    ```
- create hive table
    ```
    create table weather_data (
        weather_date string,
        station_id int,
        temp int
    )
    ```
- see tables
    ```
    show tables;
    ```
- describe table;
    ```
    describe weather_data;
    ```
- loading data
    ```
    load data loacal inpath '/path' overwrite into table weather_data;
    ```
- select
    ```
    select * from weather_data limit 5;
    ```
- creating an external table
    ```
    create external table weather_external (
        weather_date string,
        station_id int,
        temp int
    )
    row format delimited fields terminated by '\t'
    location '/path/data';
    ```
- partitioning 
    ```
    create tables weather_p (...)
    partitioned by (year string)
    row format delimited
    fields terminated be '\t';
    ```
    ```
    alter table weather_p add partition (year='1990');
    ```
- bucketing
    ```
    create table weather_b (...)
    clustered by (station_id) into 4 buckets
    row format delimited
    fields terminated by '\t';
    ```
- altering
    ```
    alter table weather_data add columns (humidity int);
    ```
- bropping
    ```
    drop table weather_data;
    ```
- querying
    ```
    select * form records;
    ```
    ```
    select year, MAX(temp) from records where temperature != 9999 and quality in (0,1,4,5,9) group by year;
    ```
    ```
    select min(temp) from weather_data;
    ```
- sorting
    ```
    select * from weather_data order by temp desc;
    ```
- aggregating
    ```
    select substr(date, 7, 4) as year, avg(temp) from weather_data group by year;
    ```
- joins
    ```
    select w.weather_date, w.temperature, s.location from weather_data w join station_info s on w.station_id = s.station_id;
    ```
- subqueries
    ```
    select * from weather where temp = (select max(temp) from weather)
    ```
- views
    ```
    create view temp as select station, avg(temp) as avg from weather group by station;
    select * from temp_summary;
    ```


## Hive UDF

- python file
    ```python
    for l in sys.stdin:
        print(l.strip().strip('.,!?'))
    ```
- executable 
    ```
    chmod +x /path
    ```
- start hive
    ```
    $HIVE_HOME
    hive
    ```
- add py file
    ```
    add file /path
    ```
- list
    ```
    list file;
    ```
- show table
    ```
    show tables;
    ```
- create table
    ```
    create table my (text string);
    ```
- describe table
    ```
    describe table;
    ```
- load data
    ```
    load data local inpath 'path' into table table;
    ```
    ```
    select * from table;
    OK 
    Hello World!!! 
    ,Data Processing., 
    Big Data!!  
    Python,Hadoop!! 
    ```
- run udf
    ```
    select transform (text_column)
    using 'python3 strip_udf.py'
    as (cleaned_text)
    from my_table;
    ```

## Spark word count
- txt file
    ```
    one two three
    four one two
    five three four
    six seven one
    two four five
    ```
- scala
    ```
    cd spark/bin
    ./spark-shell
    ```
    ```scala
    val dt = sparkContent.textFile("file:///home/path.txt")
    val dfm = dt.flatMap(x=>x.split(" ")).map(x=>(x, 1))
    val dr = dfm.reduceByKey((a,b)=>a+b)
    dr.collect()
    dr.toDF("word", "count").show()
    ```
- pyspark
    ```
    cd spark/bin
    ./pyspark
    ```
    ```py
    dt = sc.textFile('/path')
    df
    dfm = dt.flatMap(lambda x:x.split(' ')).map(lambda w: (w,1))
    dr = dfm.reduceByKey(lambda a, b: a+b)
    dr.collect()
    ```
## Spark Sql
- start spark sql
    ```
    cd spark-sql/bin
    ./spark-sql
    ```
- Table Creation from JSON File
    ```
    CREATE TABLE flights (
        DEST_COUNTRY_NAME STRING,
        ORIGIN_COUNTRY_NAME STRING,
        count LONG
    )
    USING JSON
    OPTIONS (path '/home/hadoop/Desktop/SparkSQLrmm/2015-summary.json');
    ```

    ```
    DESCRIBE TABLE flights;
    ```
    ```
    SELECT * FROM flights;
    ```
- Conditional Logic with CASE
    ```
    SELECT 
        CASE 
            WHEN DEST_COUNTRY_NAME = 'United States' THEN 1
            WHEN DEST_COUNTRY_NAME = 'Egypt' THEN 0
            ELSE -1 
        END 
    FROM flights;
    ```
- Using ARRAY
    ```
    SELECT DEST_COUNTRY_NAME, ARRAY(1, 2, 3) FROM flights;
    ```
- Aggregation

    ```
    SELECT dest_country_name 
    FROM flights 
    GROUP BY dest_country_name 
    ORDER BY sum(count) DESC 
    LIMIT 5;
    ```
- Aggregation + Sorting + Limiting
    ```
    SELECT dest_country_name 
    FROM flights 
    GROUP BY dest_country_name 
    ORDER BY sum(count) DESC 
    LIMIT 5;
    ```
- subqueries
    ```
    SELECT * 
    FROM flights 
    WHERE origin_country_name IN (
    SELECT dest_country_name 
    FROM flights 
    GROUP BY dest_country_name 
    ORDER BY sum(count) DESC 
    LIMIT 5
    );
    ```
    ```
    SELECT * 
    FROM flights f1 
    WHERE EXISTS (
    SELECT 1 FROM flights f2 
    WHERE f1.dest_country_name = f2.origin_country_name
    ) 
    AND EXISTS (
    SELECT 1 FROM flights f2 
    WHERE f2.dest_country_name = f1.origin_country_name
    );
    ```
- Function Exploration
    ```
    SHOW FUNCTIONS;
    ```
    ```
    SHOW SYSTEM FUNCTIONS;
    ```
    ```
    SHOW USER FUNCTIONS;
    ```
    ```
    SHOW FUNCTIONS "s*";
    ```
    ```
    SHOW FUNCTIONS LIKE "collect*";
    ```

## Spark Streaming
- python code
    ```py
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg
    from pyspark.sql.types import StructType, StructField, IntergerType, DoubleType
    
    sp=SparkSession.builder.appName('t1').config('spark.ui.showConsoleProgess','false').getOrCreate()
    sp.sparkContext.setLogLevel('ERROR')
    sm = StructField([
        ('cid', IntegerType(), True),
        ('temp', DoubleType(), True)
    ])
    df=sp.readStream.format('csv').option('header', 'true').schema(sm).load('folder')
    pdf=df.groupBy('cid').agg(avg('temp').alias('avg'))
    qy=pdf.writeStream.outputMode('complete').format('console').start()
    qy.awaitTermination()
    ```
- command
    ```
    cd hadoop/bin
    ./spark-submit /path.py
    ```
- csv files
    ```
    cid,temp
    1,35.6
    2,42.5
    3,42.3
    1,76.1
    2,12.8
    ```
    ```
    cid,temp
    1,53.6
    2,24.5
    4,83.1
    5,96.4
    ```
## Kafka
- check java version
    ```
    java -version
    ```
- download  kafka
- unzip
    ```
    tar -xvzf kafka_2.13-3.6.1.tgz
    ```
- format
    ```
    bin/kafka-storage.sh format -t $(bin/kafka-storage.sh random-uuid) -c config/kraft/server.properties
    ```
- start server
    ```
    bin/kafka-server-start.sh config/kraft/server.properties
    ```
- create topic
    ```
    bin/kafka-topics.sh --create --topic quickstart-events --bootstrap-server localhost:9092
    ```
- start producer
    ```
    bin/kafka-console-producer.sh --topic quickstart-events --bootstrap-server localhost:9092
    ```
- start consumer
    ```
    bin/kafka-console-consumer.sh --topic quickstart-events --from-beginning --bootstrap-server localhost:9092
    ```