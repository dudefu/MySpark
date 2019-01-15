package hnbian.sparkstreaming

import org.apache.spark.SparkConf

/**
  * Created by hnbia on 2017/4/4.
  */
object SparkStreaming_saveAdTestFile {
  def main(args: Array[String]): Unit = {
    //windows 上运行代码
    val conf = new SparkConf().setMaster("local[2]").setAppName("WordCount")
    //集群运行
    //val conf = new SparkConf().setMaster("spark://master1:7077").setAppName("WordCount")
    val ssc = new StreamingContext(conf,Seconds(5))
    ///val lines = ssc.socketTextStream("localhost",8888)
    val lines = ssc.socketTextStream("master1",9999)
    val words = lines.flatMap(_.split(" "))

    val pairs = words.map((_,1))
    val wordCounts = pairs.reduceByKey(_+_)
    wordCounts.print()
    //wordCounts.saveAsTextFiles("d:\\SparkStreaming_saveAdTestFile")
    ssc.start()
    ssc.awaitTermination()

  }
}
