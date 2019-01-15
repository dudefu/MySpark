package hnbian.sparkcore.sharedvariables

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by admin on 2017/4/6.
  * 延迟计算时，累加器重复计算异常异常
  */
object AccumulatorDeno extends Serializable {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[2]").setAppName("AccumulatorDeno")
    val sc = new SparkContext(conf)

    val rdd = sc.makeRDD(1 to 3)
    rdd.persist()
    val sum = sc.longAccumulator("sum");
    val rdd2 = rdd.map(e => {
      sum.add(1)
      e
    })
    rdd2.collect()

    //rdd2.cache.count
    // 解决延迟计算时 累加器重复计算的问题

    println("累加器值为：" + sum.value) ////此时accum的值为3，是我们要的结果

    rdd2.foreach(println) //继续操作，查看刚才变动的数据,foreach也是action操作

    //上个步骤没有进行累计器操作，可是累加器此时的结果已经是10了
    //这并不是我们想要的结果
    println("累加器值2为：" + sum.value)

    //解决办法
    //使用cache缓存数据，切断依赖。

    //rdd2.cache.count
  }
}
