package hnbian.sparkstreaming

import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

/**
  * Created by hnbia on 2017/4/4.
  */
object SparkStreaming_Window {
    def main(args: Array[String]): Unit = {
      println("开始 流处理 。 。 。")
      val conf = new SparkConf().setAppName("SparkStreamingDemo1").setMaster("local[2]")
      val ssc = new StreamingContext(conf,Seconds(5))

      //设置保存点 用于故障恢复
      //ssc.checkpoint("/tmp/spark/checkpoint")
      ssc.checkpoint("D:\\spark")

      //创建离散流，
      val streamLines = ssc.socketTextStream("master1", 9999,StorageLevel.MEMORY_AND_DISK_SER)
      // 按照窗口进行消息计数 以及打印
      streamLines
        .flatMap(_.split(" "))
        .map((_,1))
        .reduceByKeyAndWindow((a:Int,b:Int)=>{a+b},Seconds(15),Seconds(10))
        .print()
      //streamLines.reduceByWindow((_+_),Seconds(15),Seconds(10))
      // 启动流计算
      ssc.start()
      // 等待结束
      ssc.awaitTermination()
    }


}
