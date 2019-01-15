package hnbian.sparkstreaming

import org.apache.spark.SparkConf

/**
  * Created by hnbia on 2017/4/4.
  * spark streaming -kafka
  *
  */
object SparkStreaming_Kafka {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SparkStreaming_Kafka").setMaster("local[2]")
    val ssc = new StreamingContext(conf,Seconds(5))
    //制定kafka的broker 信息以及消费者信息
    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> "master1:9092,master2:9092,slave1:9092",
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "mygroupkafka",
      "auto.offset.reset" -> "latest", //最新
      "enable.auto.commit" -> (false: java.lang.Boolean)
    )
    //topic 集合 可以指定多个topic
    val topics = Array("test")
    val stream = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    )

    stream.map(record => (record.key, record.value)).print()
    ssc.start()
    ssc.awaitTermination()
  }
}
