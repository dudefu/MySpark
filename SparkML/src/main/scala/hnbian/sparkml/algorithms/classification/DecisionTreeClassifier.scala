package hnbian.spark.ml.algorithms.classification


import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import utils.FileUtils

/**
  * @author hnbian
  *         @ Description 决策树分类代码示例
  *         @ Date 2019/1/4 15:30
  **/
object DecisionTreeClassifier extends App {

  val spark = SparkUtils.getSparkSession("DecisionTree", 4)

  val filePath = FileUtils.getFilePath("iris.txt")
  println(filePath)

  import spark.implicits._

  val data = spark.sparkContext
    .textFile(filePath)
    .map(_.split(","))
    .map(p => Iris(Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble), p(4).toString())).toDF()

  data.show(false)
  /**
    * +-----------------+-----------+
    * |features         |label      |
    * +-----------------+-----------+
    * |[5.1,3.5,1.4,0.2]|Iris-setosa|
    * |[4.9,3.0,1.4,0.2]|Iris-setosa|
    * |[4.7,3.2,1.3,0.2]|Iris-setosa|
    * |[4.6,3.1,1.5,0.2]|Iris-setosa|
    * |[5.0,3.6,1.4,0.2]|Iris-setosa|
    * |[5.4,3.9,1.7,0.4]|Iris-setosa|
    * |[4.6,3.4,1.4,0.3]|Iris-setosa|
    * +-----------------+-----------+
    */
  data.map(t => t(1) + ":" + t(0)).collect().foreach(println)
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

  //接下来，我们把数据集随机分成训练集和测试集，其中训练集占70%。
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  //导入所需要的包
  import org.apache.spark.ml.classification.DecisionTreeClassificationModel
  import org.apache.spark.ml.classification.DecisionTreeClassifier
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

  //训练决策树模型,这里我们可以通过setter的方法来设置决策树的参数，
  // 也可以用ParamMap来设置（具体的可以查看spark mllib的官网）。
  // 具体的可以设置的参数可以通过explainParams()来获取。
  val dtClassifier = new DecisionTreeClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")

  //在pipeline中进行设置
  val pipelinedClassifier = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, dtClassifier, labelConverter))
  //训练决策树模型
  val modelClassifier = pipelinedClassifier.fit(trainingData)
  //进行预测
  val predictionsClassifier = modelClassifier.transform(testData)
  //查看预测结果
  predictionsClassifier.select("predictedLabel", "label", "features").show(20)

  //模型评估
  val evaluatorClassifier = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluatorClassifier.evaluate(predictionsClassifier)

  println(s"准确率 = ${accuracy * 100}%")
  //准确率 = 94.87179487179486%
  println(s"Test Error = ${(1.0 - accuracy)}")
  //Test Error = 0.0625
  val treeModelClassifier = modelClassifier.stages(2).asInstanceOf[DecisionTreeClassificationModel]
  // 查看决策树
  println("Learned classification tree model:\n" + treeModelClassifier.toDebugString)
  /**
    * DecisionTreeClassificationModel (uid=dtc_24f632ec622a) of depth 5 with 15 nodes
    * If (feature 2 <= 2.5999999999999996)
    * Predict: 2.0
    * Else (feature 2 > 2.5999999999999996)
    * If (feature 2 <= 5.05)
    * If (feature 3 <= 1.65)
    * Predict: 0.0
    * Else (feature 3 > 1.65)
    * If (feature 0 <= 6.05)
    * If (feature 1 <= 3.05)
    * Predict: 1.0
    * Else (feature 1 > 3.05)
    * Predict: 0.0
    * Else (feature 0 > 6.05)
    * Predict: 0.0
    * Else (feature 2 > 5.05)
    * If (feature 3 <= 1.65)
    * If (feature 0 <= 6.05)
    * Predict: 0.0
    * Else (feature 0 > 6.05)
    * Predict: 1.0
    * Else (feature 3 > 1.65)
    * Predict: 1.0
    */
  spark.stop()
}

case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)

