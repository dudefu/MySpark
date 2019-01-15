package hnbian.sparkstreaming

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

/**
  * Created by hnbia on 2017/4/5.
  */
object SparkStreaming_Window_DataFrame {
  def main(args: Array[String]): Unit = {
    def main(args: Array[String]): Unit = {
      println("开始 流处理 。 。 。")
      val conf = new SparkConf().setAppName("SparkStreamingDemo1").setMaster("local[2]")
      val ssc = new StreamingContext(conf,Seconds(5))

      //设置保存点 用于故障恢复
      //ssc.checkpoint("/tmp/spark/checkpoint")
      ssc.checkpoint("D:\\spark")

      //创建离散流，
      val streamLines = ssc.socketTextStream("master1", 9999,StorageLevel.MEMORY_AND_DISK_SER)
      streamLines
        .window(Seconds(6000),Seconds(10))
        .flatMap(_.split(" "))
        .foreachRDD(rdd=>{
          val sess = SparkSession.builder().config(rdd.sparkContext.getConf).getOrCreate()
          val df = rdd.toDF("word")
          df.createGlobalTempView("words")
          sess.sql("select word,count(word) as c from words group by word order by c desc").show()
        })
      // 启动流计算
      ssc.start()
      // 等待结束
      ssc.awaitTermination()
    }

  }
}
