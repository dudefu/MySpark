package hnbian.sparkcore.sharedvariables

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by admin on 2017/4/6.
  * 自定义累加器测试类
  */
object TestMyAccumulator {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("AccumulatorDeno")
    val sc = new SparkContext(conf)
    val accumulator = new MyAccumulatorV2()
    sc.register(accumulator)

    val rdd = sc.makeRDD(Array("tom","curry","green","curry","green"))
    rdd.map(e=>{
      println(e)
      accumulator.add(e)
    }).collect()
    println("累加器的值为："+accumulator.value)
  }
}
