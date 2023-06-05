#!/bin/bash
source .env
$KAFKA_PATH/bin/zookeeper-server-start.sh -daemon $KAFKA_PATH/config/zookeeper.properties
$KAFKA_PATH/bin/kafka-server-start.sh -daemon $KAFKA_PATH/config/server.properties
$KAFKA_PATH/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 3 --topic transaction