package hnbian.spark.utils

import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}


/**
  * @author hnbian
  * @ Description 获取SparkSession
  * @ Date 2018/12/27 15:43
  **/
object SparkUtils {

  def getSparkSession(appName:String,workers:Int): SparkSession ={
    val conf = new SparkConf().setAppName(appName)
    //设置master local[worker] 指定本地模式开启模拟worker线程数
    conf.setMaster(s"local[${workers}]")
    //创建sparkContext文件
    val sc = new SparkContext(conf)
    SparkSession.builder().getOrCreate()
  }
}
