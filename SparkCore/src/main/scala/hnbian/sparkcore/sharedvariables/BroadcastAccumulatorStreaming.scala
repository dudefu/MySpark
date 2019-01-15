package hnbian.sparkcore.sharedvariables

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.util.LongAccumulator
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by admin on 2017/4/6.
  * spark-streaming + broadcast + accumulotor 示例
  */
object BroadcastAccumulatorStreaming {
  /**
    * 声明一个广播变量
    */
  private var broadcastList: Broadcast[List[String]] = _
  /**
    * 声明一个累加器
    */
  private var accumulator: LongAccumulator = _

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("broadcasttest")
    val sc = new SparkContext(sparkConf)
    /**
      * duration是ms
      */
    //val ssc = new StreamingContext(sc,Duration(2000))

    /*val ssc = new StreamingContext(sc, Seconds(5))


    broadcastList = ssc.sparkContext.broadcast(List("Hadoop", "Spark"))

    //accumulator= ssc.sparkContext.accumulator(0,"broadcasttest")
    accumulator = ssc.sparkContext.longAccumulator("broadcast_test")

    //获取数据
    val lines = ssc.socketTextStream("localhost", 9999)*/

    /**
      * 拿到数据后 怎么处理！
      *
      * 1.flatmap把行分割成词。
      * 2.map把词变成tuple(word,1)
      * 3.reducebykey累加value
      * (4.sortBykey排名)
      * 4.进行过滤。 value是否在累加器中。
      * 5.打印显示。
      */

    /*val words = lines.flatMap(line => line.split(" "))

    val wordpair = words.map(word => (word, 1))

    wordpair.filter(record => {
      broadcastList.value.contains(record._1)
    })

    val pair = wordpair.reduceByKey(_ + _)*/

    /**
      * 这步为什么要先foreachRDD？
      * 因为这个pair 是PairDStream<String, Integer>
      * 进行foreachRDD是为了？
      */

    /*val filtedpair = pair.filter(record => {
      if (broadcastList.value.contains(record._1)) {
        accumulator.add(record._2)
        true
      } else {
        false
      }
    }
    ).print
    println("累加器的值" + accumulator.value)
    pair.print()
    ssc.start()
    ssc.awaitTermination()*/
  }
}
