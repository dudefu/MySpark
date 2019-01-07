package hnbian.spark.ml.algorithms.classification.classnews

import utils.SparkUtils

/**
  * @author hnbian
  *         @ Description
  *         @ Date 2019/1/7 10:41
  **/
object LogisticRegressionDemo extends App {


  //val spark = SparkUtils.getSparkSession("LogisticRegressionDemo", 4)
  //获得数据路径地址

 val filePath =  System.getProperty("user.dir")+"/src/main/resources/data/classnews/train"
  println(filePath)

}
