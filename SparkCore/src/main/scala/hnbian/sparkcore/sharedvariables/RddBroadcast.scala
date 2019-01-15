package hnbian.sparkcore.sharedvariables

import java.io.FileWriter

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hnbia on 2017/4/2.
  * 广播变量 和 累加器
  */
object RddBroadcast {
  def main(args: Array[String]) {
    //设置master local[4] 指定本地模式开启模拟worker线程数
    val conf = new SparkConf().setAppName("RDDMapPartitions").setMaster("local[4]")

    //创建sparkContext文件
    val sc = new SparkContext(conf)
    /*val bc = sc.broadcast("hello world")
    println(bc)
    println(bc.value)*/

    val bc = sc.broadcast("c")
    println(bc.value)
    val rdd = sc.makeRDD(1 to 5)
    var count = 1;
    rdd.map(e=>{
      val f = new FileWriter("/opt/appl/broadcast.txt",true)
      while(bc.value.eq("c")){
        count +=1
        f.write(count)
      }
      e
    })

    //累加器
    val sum = sc.longAccumulator("sum")
    val rdd1 = sc.makeRDD(1 to 5)
    rdd1.map(e=>{
      sum.add(e*3)
    }).collect();
    println("累加器值为："+sum.value)
  }
}
