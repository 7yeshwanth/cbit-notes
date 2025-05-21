
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