package hnbian.sparkml.algorithms.regression

import hnbian.spark.ml.algorithms.classification.DecisionTreeClassifier.{data, labelIndexer}
import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import utils.FileUtils

/**
  * @author hnbian 2019/1/17 17:05
  *         决策树回归代码示例
  */
object DecisionTreeRegressor extends App {
  val spark = SparkUtils.getSparkSession("DecisionTreeRegressor", 4)

  val filePath = FileUtils.getFilePath("iris.txt")
  println(filePath)

  import spark.implicits._

  //加载数据
  val data = spark.sparkContext
    .textFile(filePath)
    .map(_.split(","))
    .map(p => Iris(Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble), p(4).toString())).toDF()

  //展示数据
  data.show(false)
  /**
    * +-----------------+-----------+
    * |features         |label      |
    * +-----------------+-----------+
    * |[5.1,3.5,1.4,0.2]|Iris-setosa|
    * |[4.9,3.0,1.4,0.2]|Iris-setosa|
    * |[4.7,3.2,1.3,0.2]|Iris-setosa|
    * |[4.6,3.1,1.5,0.2]|Iris-setosa|
    * +-----------------+-----------+
    */
  //定义决策树实例
  val dtRegressor = new DecisionTreeRegressor()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")

  //分别获取标签列和特征列，进行索引，并进行了重命名。
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)
  //labelIndexer.transform(data)

  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(data)

  //这里我们设置一个labelConverter，目的是把预测的类别重新转化成字符型的。
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  //在pipeline中进行设置
  val pipelineRegressor = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dtRegressor, labelConverter))

  //接下来，我们把数据集随机分成训练集和测试集，其中训练集占70%。
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
  //训练决策树模型
  val modelRegressor = pipelineRegressor.fit(trainingData)
  //预测
  val predictionsRegressor = modelRegressor.transform(testData)
  //展示预测结果
  predictionsRegressor.show()
  /**
    * +-----------------+---------------+------------+-----------------+----------+---------------+
    * |         features|          label|indexedLabel|  indexedFeatures|prediction| predictedLabel|
    * +-----------------+---------------+------------+-----------------+----------+---------------+
    * |[4.4,3.0,1.3,0.2]|    Iris-setosa|         2.0|[4.4,3.0,1.3,0.2]|       2.0|    Iris-setosa|
    * |[4.6,3.1,1.5,0.2]|    Iris-setosa|         2.0|[4.6,3.1,1.5,0.2]|       2.0|    Iris-setosa|
    * |[4.6,3.2,1.4,0.2]|    Iris-setosa|         2.0|[4.6,3.2,1.4,0.2]|       2.0|    Iris-setosa|
    * |[4.6,3.4,1.4,0.3]|    Iris-setosa|         2.0|[4.6,3.4,1.4,0.3]|       2.0|    Iris-setosa|
    * |[4.8,3.4,1.9,0.2]|    Iris-setosa|         2.0|[4.8,3.4,1.9,0.2]|       2.0|    Iris-setosa|
    * |[4.9,3.1,1.5,0.1]|    Iris-setosa|         2.0|[4.9,3.1,1.5,0.1]|       2.0|    Iris-setosa|
    * +-----------------+---------------+------------+-----------------+----------+---------------+
    */

  //模型评估
  val evaluatorRegressor = new RegressionEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("rmse")

  //获取标准误差
  val rmse = evaluatorRegressor.evaluate(predictionsRegressor)

  //打印标准误差
  println("Root Mean Squared Error (RMSE) on test data = " + rmse)
  //Root Mean Squared Error (RMSE) on test data = 0.1414213562373095

  val treeModelRegressor = modelRegressor.stages(2).asInstanceOf[DecisionTreeRegressionModel]
  //打印决策树
  println("Learned regression tree model:\n" + treeModelRegressor.toDebugString)
  /**
    *Learned regression tree model:
    *DecisionTreeRegressionModel (uid=dtr_a1688d438ba6) of depth 4 with 11 nodes
    *  If (feature 2 <= 2.45)
    *   Predict: 2.0
    *  Else (feature 2 > 2.45)
    *   If (feature 3 <= 1.75)
    *    If (feature 2 <= 4.95)
    *     If (feature 0 <= 4.95)
    *      Predict: 1.0
    *     Else (feature 0 > 4.95)
    *      Predict: 0.0
    *    Else (feature 2 > 4.95)
    *     If (feature 3 <= 1.65)
    *      Predict: 1.0
    *     Else (feature 3 > 1.65)
    *      Predict: 0.0
    *   Else (feature 3 > 1.75)
    *    Predict: 1.0
    */
}

case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)
