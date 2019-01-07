package hnbian.spark.ml.algorithms.classification.classnews

import hnbian.spark.algorithms.classification.classnews.Preprocessor
import utils.SparkUtils


/**
  * @author hnbian
  *         @ Description 使用逻辑回归实现对新闻多分类的逻辑
  *         @ Date 2019/1/7 10:41
  **/
object LogisticRegressionDemo extends App {

  val files = Array("culture.txt", "", "", "")

  val spark = SparkUtils.getSparkSession("LogisticRegressionDemo", 4)
  //获得数据路径地址 windows 运行时后面需要加*号 不然会报错
  val filePath = System.getProperty("user.dir") + "/SparkML/src/main/resources/data/classnews/train/*"
  println(filePath)
  val p = new Preprocessor()
  val textDF = p.clean(filePath, spark)
  val model = p.indexrize(textDF)
}
