package hnbian.sparkstreaming

import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel

/**
  * Created by hnbia on 2017/4/4.
  * 首先启动nc 服务器 nc -lp 9999
  */
object SparkStreamingDemo1 {
  def main(args: Array[String]): Unit = {
    println("开始 流处理 。 。 。")
    val conf = new SparkConf().setAppName("print words").setMaster("local[2]")
    val ssc = new StreamingContext(conf,Seconds(5))
    //设置保存点 用于故障恢复
    //ssc.checkpoint("/tmp/spark/checkpoint")
    ssc.checkpoint("D:\\spark")

    println("Stream processing")

    //创建离散流，
    val streamLines = ssc.socketTextStream("master1", 9999,StorageLevel.MEMORY_AND_DISK_SER)
    // 对流进行过滤 包含“ERROR” 字样的文本
    //打印流
    streamLines.print()
    println("Stream processing logic end")
    // 启动流计算
    ssc.start()
    // 等待结束
    ssc.awaitTermination()
  }
}
