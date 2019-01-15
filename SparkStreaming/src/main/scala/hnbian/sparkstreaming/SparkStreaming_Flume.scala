package hnbian.sparkstreaming

import org.apache.spark.SparkConf

/**
  * Created by hnbia on 2017/4/4.
  */
object SparkStreaming_Flume {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SparkStreaming_Flume").setMaster("local[2]")
    val ssc = new StreamingContext(conf, Seconds(5))
    val flumeStream = FlumeUtils.createStream(ssc, "master1", 9999)
    flumeStream.map(e => e.event.getBody.toString)
    println(flumeStream)
  }
}
