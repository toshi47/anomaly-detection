from kafka import KafkaProducer    
import json
import pandas
from decouple import config

DATA_PATH=config('DATA_PATH')
NETWORK=config('NETWORK')
KAFKA_SERVER_PORT=config('KAFKA_SERVER_PORT')
TOPIC_NAME=config('TOPIC_NAME')

def get_random_value():
    df=pandas.read_csv(DATA_PATH)
    row=df.sample()
    dict=row.to_dict('records')[0]
    return dict
 

if __name__ == "__main__":
    producer = KafkaProducer(bootstrap_servers=[NETWORK+':'+KAFKA_SERVER_PORT],
                                value_serializer=lambda x:json.dumps(x).encode('utf-8'),
                                compression_type='gzip')
    my_topic = TOPIC_NAME
    data = get_random_value()

    try:
        future = producer.send(topic = my_topic, value = data)
        record_metadata = future.get(timeout=10)
        print('--> The message has been sent to a topic: {}, partition: {}, offset: {}' .format(record_metadata.topic,record_metadata.partition, record_metadata.offset ))   
                                
    except Exception as e:
        print('--> It seems an Error occurred: {}'.format(e))

    finally:
        producer.flush()
