import findspark
from pyspark.sql import SparkSession 
from pyspark.sql.functions import expr, from_json, col, concat
from pyspark.sql.types import *
from decouple import config
import os

SPARK_PATH=config('SPARK_PATH')
SPARK_CHECKPOINT_PATH=config('SPARK_CHECKPOINT_PATH')
NETWORK=config('NETWORK')
KAFKA_SERVER_PORT=config('KAFKA_SERVER_PORT')
TOPIC_NAME=config('TOPIC_NAME')

findspark.init(SPARK_PATH)
os.environ['PYSPARK_SUBMIT_ARGS'] = config('SPARK_SUBMIT_ARGS')



if __name__ == "__main__":
  spark = (SparkSession
          .builder
          .appName("consumer_structured_streaming_test")  
          .getOrCreate())

  columns={'dt':IntegerType(),'switch':IntegerType(),'src':StringType(),'dst':StringType(),'pktcount':IntegerType(),'bytecount':IntegerType(),\
          'dur':IntegerType(),'dur_nsec':LongType(),'tot_dur':FloatType(),'flows':IntegerType(),'packetins':IntegerType(),'pktperflow':IntegerType(),\
          'byteperflow':IntegerType(),'pktrate':IntegerType(),'Pairflow':IntegerType(),'Protocol':StringType(),'port_no':IntegerType(),'tx_bytes':IntegerType(),\
          'rx_bytes':IntegerType(),'tx_kbps':IntegerType(),'rx_kbps':FloatType(),'tot_kbps':FloatType(),'label':IntegerType()}
  
  fields_lst=[]
  
  for key in columns.keys():
      fields_lst.append(StructField(key,columns[key]))

  schema = StructType(fields_lst)

  # Subscribe to 1 topic defaults to the earliest and latest offsets
  df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", NETWORK+':'+KAFKA_SERVER_PORT) \
    .option("subscribe", TOPIC_NAME) \
    .load()


  value_df = df.select(from_json(col("value").cast("string"),schema).alias("value"))

  query=value_df.writeStream.format("console") \
              .option("truncate", "false") \
              .outputMode("append") \
              .option("checkpointLocation", SPARK_CHECKPOINT_PATH) \
              .trigger(processingTime='5 seconds') \
              .start() \
              .awaitTermination()