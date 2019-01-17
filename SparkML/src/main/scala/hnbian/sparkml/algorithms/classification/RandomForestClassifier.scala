package hnbian.sparkml.algorithms.classification

import utils.FileUtils
import hnbian.spark.utils.SparkUtils
import hnbian.sparkml.utils.Evaluations
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

/**
  * @author hnbian
  * @ Description  随机森林分类代码示例
  * @ Date 2019/1/10 16:18
  **/
object RandomForestClassifier extends App {


  val spark = SparkUtils.getSparkSession("RandomForestClassifier", 4)

  val filePath = FileUtils.getFilePath("sample_libsvm_data.txt")

  val data = spark.read.format("libsvm").load(filePath)

  //打印数据
  data.show()
  /**
    * +-----+--------------------+
    * |label|            features|
    * +-----+--------------------+
    * |  1.0|(692,[125,126,153...|
    * |  1.0|(692,[127,128,154...|
    * |  1.0|(692,[154,155,156...|
    * |  1.0|(692,[152,153,154...|
    * |  0.0|(692,[127,128,129...|
    * +-----+--------------------+
    */
  //标签转为向量索引
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)

  //自动识别分类特征，并为它们编制索引。
  //设置maxCategories，所以具有> 4个不同值的特征被视为连续的。
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(data)

  // 将数据分成训练和测试集（30％用于测试）。
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
  // 设置随机森林模型
  val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")
    .setNumTrees(3) //分成3棵树

  // 将索引标签转换回原始标签。
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // 将每个模型添加到工作流中
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

  // 训练模型
  val model = pipeline.fit(trainingData)

  // 进行预测
  val predictions = model.transform(testData)
  predictions.show(5)
  /**
    * +-----+--------------------+------------+--------------------+-------------+-----------+----------+--------------+
    * |label|            features|indexedLabel|     indexedFeatures|rawPrediction|probability|prediction|predictedLabel|
    * +-----+--------------------+------------+--------------------+-------------+-----------+----------+--------------+
    * |  0.0|(692,[100,101,102...|         1.0|(692,[100,101,102...|    [5.0,5.0]|  [0.5,0.5]|       0.0|           1.0|
    * |  0.0|(692,[121,122,123...|         1.0|(692,[121,122,123...|   [0.0,10.0]|  [0.0,1.0]|       1.0|           0.0|
    * |  0.0|(692,[124,125,126...|         1.0|(692,[124,125,126...|   [0.0,10.0]|  [0.0,1.0]|       1.0|           0.0|
    * |  0.0|(692,[125,126,127...|         1.0|(692,[125,126,127...|    [1.0,9.0]|  [0.1,0.9]|       1.0|           0.0|
    * |  0.0|(692,[126,127,128...|         1.0|(692,[126,127,128...|    [2.0,8.0]|  [0.2,0.8]|       1.0|           0.0|
    * +-----+--------------------+------------+--------------------+-------------+-----------+----------+--------------+
    */


  //模型评估
  val (accuracy, precision, recall, f1) = Evaluations.multiClassEvaluate(predictions)
  println("\n\n========= 评估结果 ==========")
  println(s"\n准确率：$accuracy")
  println(s"加权精确率：$precision")
  println(s"加权召回率：$recall")
  println(s"F1值：$f1")
  /**
    * 准确率：0.96875
    * 加权精确率：0.9711538461538461
    * 加权召回率：0.96875
    * F1值：0.9689743589743589
    */
  val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
  println("Learned classification forest model:\n" + rfModel.toDebugString)
  /**
    * Learned classification forest model:
    * RandomForestClassificationModel (uid=rfc_82f005014192) with 3 trees
    * Tree 0 (weight 1.0):
    * If (feature 512 <= 8.0)
    * If (feature 454 <= 12.0)
    * If (feature 486 <= 212.0)
    * Predict: 0.0
    * Else (feature 486 > 212.0)
    * Predict: 1.0
    * Else (feature 454 > 12.0)
    * Predict: 1.0
    * Else (feature 512 > 8.0)
    * Predict: 1.0
    *
    * Tree 1 (weight 1.0):
    * If (feature 462 <= 63.0)
    * If (feature 492 <= 190.5)
    * Predict: 1.0
    * Else (feature 492 > 190.5)
    * Predict: 0.0
    * Else (feature 462 > 63.0)
    * Predict: 0.0
    *
    * Tree 2 (weight 1.0):
    * If (feature 524 <= 21.0)
    * If (feature 429 <= 7.0)
    * Predict: 0.0
    * Else (feature 429 > 7.0)
    * Predict: 1.0
    * Else (feature 524 > 21.0)
    * Predict: 1.0
    */
}
