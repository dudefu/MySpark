package hnbian.sparkstreaming

import org.apache.spark.SparkConf

/**
  * Created by hnbia on 2017/4/4.
  */
object FileStreamingDemo {
  def main(args: Array[String]): Unit = {





    val conf = new SparkConf().setAppName("FileStreamDemo").setMaster("local[2]")
    val ssc = new StreamingContext(conf,Seconds(5))
    val stream = ssc.textFileStream("d:/data")
    stream.map(f=>{
      println(f + "----")
    })
    stream.print()
    ssc.start()
    ssc.awaitTermination()


  }
}
