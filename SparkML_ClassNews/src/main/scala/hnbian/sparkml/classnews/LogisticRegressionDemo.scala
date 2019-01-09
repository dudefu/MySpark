package hnbian.sparkml.classnews

import hnbian.spark.utils.SparkUtils
import hnbian.sparkml.classnews.preprocess.Preprocessor
import org.slf4j.impl.Log4jLoggerFactory


/**
  * @author hnbian
  *         @ Description 使用逻辑回归实现对新闻多分类的逻辑
  *         @ Date 2019/1/7 10:41
  **/
object LogisticRegressionDemo extends App {



  val spark = SparkUtils.getSparkSession("LogisticRegressionDemo", 4)
  //获得数据路径地址 windows 运行时后面需要加*号 不然会报错
  val filePath = System.getProperty("user.dir") + "/SparkML_ClassNews/src/main/resources/train/*"
  println(filePath)

  //训练模型
  /*val preprocessor = new Preprocessor
  val trainDF = preprocessor.train(filePath, spark)*/

/*
  val textRDD = spark.sparkContext.textFile(filePath)
  println(textRDD.collect().length)
  val p = new Preprocessor()
  val textDF = p.clean(filePath, spark)
  val model = p.indexrize(textDF)*/

  //=== 预处理(清洗、标签索引化、分词、向量化)
  val preprocessor = new Preprocessor
  val trainDF = preprocessor.predict(filePath, spark)._1

  //=== 模型训练
  val lrClassifier = new LRClassifier
  lrClassifier.train(trainDF)

  spark.stop()
}
