from kafka import KafkaProducer    
import json
import pandas

def get_random_value():
    df=pandas.read_csv('kafka/dataset_sdn.csv')
    row=df.sample()
    dict=row.to_dict('records')[0]
    return dict
 

if __name__ == "__main__":
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                                value_serializer=lambda x:json.dumps(x).encode('utf-8'),
                                compression_type='gzip')
    my_topic = 'transaction'
    data = get_random_value()

    try:
        future = producer.send(topic = my_topic, value = data)
        record_metadata = future.get(timeout=10)
        print('--> The message has been sent to a topic: {}, partition: {}, offset: {}' .format(record_metadata.topic,record_metadata.partition, record_metadata.offset ))   
                                
    except Exception as e:
        print('--> It seems an Error occurred: {}'.format(e))

    finally:
        producer.flush()
