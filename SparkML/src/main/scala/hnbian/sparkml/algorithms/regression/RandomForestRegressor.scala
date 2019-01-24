package hnbian.sparkml.algorithms.regression

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import utils.FileUtils

/**
  * @author hnbian 2019/1/18 11:46
  *         随机森林回归代码示例
  */
object RandomForestRegressor extends App {
  val filePath = FileUtils.getFilePath("sample_libsvm_data.txt")
  val spark = SparkUtils.getSparkSession("RandomForestRegressor", 4)

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
  //自动识别分类特征，并为它们编制索引。
  //设置maxCategories，所以具有> 4个不同值的特征被视为连续的。
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(data)

  //将数据分成训练和测试集（30％用于测试）。
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  // Train a RandomForest model.
  val rf = new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("indexedFeatures")
    .setNumTrees(4)

  // 创建工作流并添加stage
  val pipeline = new Pipeline()
    .setStages(Array(featureIndexer, rf))

  // 训练模型
  val model = pipeline.fit(trainingData)

  // 进行预测
  val predictions = model.transform(testData)
  predictions.show(30)
  /**
    * +-----+--------------------+--------------------+----------+
    * |label|            features|     indexedFeatures|prediction|
    * +-----+--------------------+--------------------+----------+
    * |  0.0|(692,[100,101,102...|(692,[100,101,102...|       0.1|
    * |  0.0|(692,[123,124,125...|(692,[123,124,125...|       0.0|
    * |  0.0|(692,[124,125,126...|(692,[124,125,126...|       0.0|
    * +-----+--------------------+--------------------+----------+
    */
  // 选择（预测，真实标签）并计算测试错误。
  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
  //打印标准误差
  val rmse = evaluator.evaluate(predictions)
  println("Root Mean Squared Error (RMSE) on test data = " + rmse)
  //Root Mean Squared Error (RMSE) on test data = 0.19695964928958382
  val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
  println("Learned regression forest model:\n" + rfModel.toDebugString)

  /**
    * Learned regression forest model:
    * RandomForestRegressionModel (uid=rfr_54d456fb4103) with 4 trees
    *   Tree 0 (weight 1.0):
    *     If (feature 434 <= 70.5)
    *      Predict: 0.0
    *     Else (feature 434 > 70.5)
    *      Predict: 1.0
    *   Tree 1 (weight 1.0):
    *     If (feature 489 <= 1.5)
    *      Predict: 0.0
    *     Else (feature 489 > 1.5)
    *      Predict: 1.0
    *   Tree 2 (weight 1.0):
    *     If (feature 434 <= 70.5)
    *      Predict: 0.0
    *     Else (feature 434 > 70.5)
    *      Predict: 1.0
    *   Tree 3 (weight 1.0):
    *     If (feature 490 <= 29.0)
    *      Predict: 0.0
    *     Else (feature 490 > 29.0)
    *      Predict: 1.0
    */

}
