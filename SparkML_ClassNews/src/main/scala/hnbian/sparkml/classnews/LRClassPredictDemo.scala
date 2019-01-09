package hnbian.sparkml.classnews

import hnbian.spark.utils.SparkUtils
import hnbian.sparkml.classnews.preprocess.Preprocessor
import hnbian.sparkml.classnews.utils.Evaluations


/**
  * @author hnbian
  *         @ Description  数据预测与评估
  *         @ Date 2019/1/7 16:06
  **/
object LRClassPredictDemo extends App {

  val filePath = System.getProperty("user.dir") + "/SparkML_ClassNews/src/main/resources/train/*"
  val spark = SparkUtils.getSparkSession("LRClassPredictDemo", 4)

  //=== 预处理(清洗、分词、向量化)
  val preprocessor = new Preprocessor
  val (predictDF, indexModel, _) = preprocessor.predict(filePath, spark)

  //=== 模型预测
  val lrClassifier = new LRClassifier
  val predictions = lrClassifier.predict(predictDF, indexModel)

  val (accuracy,precision, recall, f1) = Evaluations.multiClassEvaluate(predictions)
  println("\n\n========= 评估结果 ==========")
  println(s"\n准确率：$accuracy")
  println(s"加权精确率：$precision")
  println(s"加权召回率：$recall")
  println(s"F1值：$f1")

  predictions.select("indexedLabel","prediction","probability").show(5, truncate = false)

  spark.stop()
}
