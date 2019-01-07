package hnbian.sparkml.classnews



import org.apache.spark.sql.Row
import hnbian.spark.utils.SparkUtils
import hnbian.sparkml.classnews.preprocess.Preprocessor
import hnbian.sparkml.classnews.utils.Evaluations


/**
  * @author hnbian
  *         @ Description
  *         @ Date 2019/1/7 16:06
  **/
object LRClassPredictDemo extends App {

  val filePath = System.getProperty("user.dir") + "/SparkML/src/main/resources/data/classnews/train/*"
  val spark = SparkUtils.getSparkSession("LRClassPredictDemo", 4)

  //=== 预处理(清洗、分词、向量化)
  val preprocessor = new Preprocessor
  val (predictDF, indexModel, _) = preprocessor.predict(filePath, spark)

  //=== 模型预测
  val lrClassifier = new LRClassifier
  val predictions = lrClassifier.predict(predictDF, indexModel)

  //=== 模型评估
  val resultRDD = predictions.select("prediction", "indexedLabel").rdd.map { case Row(prediction: Double, label: Double) => (prediction, label) }
  val (precision, recall, f1) = Evaluations.multiClassEvaluate(resultRDD)
  println("\n\n========= 评估结果 ==========")
  println(s"\n加权准确率：$precision")
  println(s"加权召回率：$recall")
  println(s"F1值：$f1")

  predictions.select("label", "predictedLabel", "content").show(100, truncate = false)

  spark.stop()
}
