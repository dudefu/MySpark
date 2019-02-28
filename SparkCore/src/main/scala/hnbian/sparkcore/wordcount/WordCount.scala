package hnbian.sparkcore.wordcount

import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author hnbian 2019/2/28 10:20
  */
object WordCount  {
  def main(args: Array[String]): Unit = {
    val filePath = "D:\\Documents\\WordCount.txt"
    //Life is a journey not the destination but the scenery along the should be and the mood at the view
    val conf = new SparkConf().setAppName("WordCount Application")
    //设置master local[4] 指定本地模式开启模拟worker线程数
    conf.setMaster("local[4]")
    //创建sparkContext文件
    val sc = new SparkContext(conf)
    val rdd1 = sc.textFile(filePath)
      .flatMap(words=>{words.split(" ")})
      .map(words=>(words,1))
      .reduceByKey((a,b)=> a + b)

    rdd1.foreach(words=>{
      println(words)
    })
  }

  /**
    * (is,1)
    * (not,1)
    * (journey,1)
    * (a,1)
    * (destination,1)
    * (be,1)
    * (Life,1)
    * (at,1)
    * (scenery,1)
    * (mood,1)
    * (should,1)
    * (along,1)
    * (but,1)
    * (and,1)
    * (view,1)
    * (the,5)
    */
}
