package hnbian.sparkml.algorithms.classification

import hnbian.spark.utils.SparkUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row
import utils.FileUtils

/**
  * @author hnbian
  *         @ Description 
  *         @ Date 2019/1/11 17:32
  **/
object OneVsRest extends App {

  val spark = SparkUtils.getSparkSession("OneVsRest", 4)
  val filePath = FileUtils.getFilePath("sample_libsvm_data.txt")
  val inputData = spark.read.format("libsvm").load(filePath)
  inputData.show(3)
  /**
    * +-----+--------------------+
    * |label|            features|
    * +-----+--------------------+
    * |  1.0|(692,[125,126,153...|
    * |  1.0|(692,[127,128,154...|
    * |  1.0|(692,[154,155,156...|
    * +-----+--------------------+
    */

  import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}

  // 生成训练/测试分组。
  val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

  // 实例化基础分类器
  val classifier = new LogisticRegression()
    .setMaxIter(10)
    .setTol(1E-6)
    .setFitIntercept(true)

  // 实例化 the One Vs Rest 分类器.
  val ovr = new OneVsRest().setClassifier(classifier)

  // 训练多类别模型
  val ovrModel = ovr.fit(train)

  // 在测试数据上评分模型。
  val predictions = ovrModel.transform(test)
  predictions.show(2)
  /**
    * +-----+--------------------+--------------------+----------+
    * |label|            features|       rawPrediction|prediction|
    * +-----+--------------------+--------------------+----------+
    * |  0.0|(692,[121,122,123...|[13.5013498158248...|       0.0|
    * |  0.0|(692,[122,123,124...|[11.4299817155731...|       0.0|
    * +-----+--------------------+--------------------+----------+
    */

  //模型评估
  val metrics = new MulticlassMetrics(
    predictions.select("prediction", "label")
      .rdd.map {
      case Row(prediction: Double, label: Double) => (prediction, label)
    }
  )

  println("\n\n========= 评估结果 ==========")
  println(s"\n准确率：${metrics.accuracy}")
  println(s"加权精确率：${metrics.weightedPrecision}")
  println(s"加权召回率：${metrics.weightedRecall}")
  println(s"F1值：${metrics.weightedFMeasure}")

  /**
    * 评估结果
    *
    * 准确率：1.0
    * 加权精确率：1.0
    * 加权召回率：1.0
    * F1值：1.0
    */

}
