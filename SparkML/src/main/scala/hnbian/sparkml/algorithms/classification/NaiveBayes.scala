package hnbian.sparkml.algorithms.classification

import hnbian.spark.utils.SparkUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row
import utils.FileUtils

/**
  * @author hnbian
  * @ Description 朴素贝叶斯代码示例
  * @ Date 2019/1/14 11:00
  **/
object NaiveBayes extends App {
  val spark = SparkUtils.getSparkSession("NaiveBayes", 4)
  val filePath = FileUtils.getFilePath("sample_libsvm_data.txt")

  val data = spark.read.format("libsvm").load(filePath)
  data.show()
  /**
    * +-----+--------------------+
    * |label|            features|
    * +-----+--------------------+
    * |  1.0|(692,[125,126,153...|
    * |  1.0|(692,[127,128,154...|
    * |  1.0|(692,[154,155,156...|
    * |  1.0|(692,[152,153,154...|
    * +-----+--------------------+
    */

  import org.apache.spark.ml.classification.NaiveBayes
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

  // 将数据分成训练和测试集（30％用于测试）
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

  // 训练朴素贝叶斯模型
  val model = new NaiveBayes()
    .fit(trainingData)

  // 选择要显示的示例行。
  val predictions = model.transform(testData)
  predictions.show()

  /**
    * +-----+--------------------+--------------------+-----------+----------+
    * |label|            features|       rawPrediction|probability|prediction|
    * +-----+--------------------+--------------------+-----------+----------+
    * |  0.0|(692,[95,96,97,12...|[-173678.60946628...|  [1.0,0.0]|       0.0|
    * |  0.0|(692,[98,99,100,1...|[-178107.24302988...|  [1.0,0.0]|       0.0|
    * |  0.0|(692,[100,101,102...|[-100020.80519087...|  [1.0,0.0]|       0.0|
    * +-----+--------------------+--------------------+-----------+----------+
    */

  //模型评估
  val metrics = new MulticlassMetrics(
    predictions.select("prediction", "label")
      .rdd.map {
      case Row(prediction: Double, label: Double) => (prediction, label)
    }
  )

  println("\n\n评估结果")
  println(s"\n准确率：${metrics.accuracy}")
  println(s"加权精确率：${metrics.weightedPrecision}")
  println(s"加权召回率：${metrics.weightedRecall}")
  println(s"F1值：${metrics.weightedFMeasure}")
  /**
    * 评估结果
    * 准确率：1.0
    * 加权精确率：1.0
    * 加权召回率：1.0
    * F1值：1.0
    */
}
