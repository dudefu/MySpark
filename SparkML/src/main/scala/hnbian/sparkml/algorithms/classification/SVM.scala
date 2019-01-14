package hnbian.sparkml.algorithms.classification

import hnbian.spark.utils.SparkUtils
import hnbian.sparkml.utils.Evaluations
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row
import utils.FileUtils

/**
  * @author hnbian
  * @ Description 支持向量机代码示例
  * @ Date 2019/1/14 15:56
  **/
object SVM extends App {
  val spark = SparkUtils.getSparkSession("SVM", 4)
  val filePath = FileUtils.getFilePath("iris.txt")

  import spark.implicits._

  //在我们的数据集中，每行被分成了5部分，前4部分是鸢尾花的4个特征，最后一部分是鸢尾花的分类。
  val data = spark.sparkContext
    .textFile(filePath)
    .map(_.split(","))
    .map(p => Iris(Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble), p(4).toString()))
    //从训练数据去掉一个分类去掉
    .filter(p => {
    p.labelCol != "Iris-versicolor"
  })
    .toDF()
  data.show()

  //分别获取标签列和特征列，进行索引，并进行了重命名。
  val labelIndexer = new StringIndexer()
    .setInputCol("labelCol")
    .setOutputCol("label")
    .fit(data)

  //从训练数据中取两个分类，把第三个分类去掉
  val labelIndexerData = labelIndexer.transform(data) //.where("label != 2")
  labelIndexerData.persist()
  labelIndexerData.show(30)

  /*labelIndexerData.groupBy("label").count().show()
  labelIndexerData.groupBy("labelCol").count().show()*/
  /**
    * +-----------------+---------------+------------+
    * |         features|          label|indexedLabel|
    * +-----------------+---------------+------------+
    * |[7.0,3.2,4.7,1.4]|Iris-versicolor|         0.0|
    * |[6.4,3.2,4.5,1.5]|Iris-versicolor|         0.0|
    * +-----------------+---------------+------------+
    */

  //接下来，我们把数据集随机分成训练集和测试集，其中训练集占70%。
  val Array(trainingData, testData) = labelIndexerData.randomSplit(Array(0.7, 0.3))
  //可线性分类SVM
  val lsvc = new LinearSVC().setRegParam(0.3).setMaxIter(100)

  val model = lsvc.fit(trainingData)
  //class org.apache.spark.ml.classification.LinearSVCModel

  println(model.getClass)
  val predictions = model.transform(testData)
  predictions.show()
  labelIndexerData.unpersist()
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
    *
    * 准确率：1.0
    * 加权精确率：1.0
    * 加权召回率：1.0
    * F1值：1.0
    */
}

case class Iris(features: org.apache.spark.ml.linalg.Vector, labelCol: String)
