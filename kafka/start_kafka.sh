#!/bin/bash
source .env
$KAFKA_PATH/bin/zookeeper-server-start.sh -daemon $KAFKA_PATH/config/zookeeper.properties
$KAFKA_PATH/bin/kafka-server-start.sh -daemon $KAFKA_PATH/config/server.properties
$KAFKA_PATH/bin/kafka-topics.sh --create --zookeeper $NETWORK:$ZOOKEEPER_SERVER_PORT --replication-factor 1 --partitions 3 --topic $TOPIC_NAME