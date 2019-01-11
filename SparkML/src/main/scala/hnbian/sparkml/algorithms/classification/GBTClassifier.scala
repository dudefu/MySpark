package hnbian.sparkml.algorithms.classification

import hnbian.spark.utils.SparkUtils
import hnbian.sparkml.utils.Evaluations
import utils.FileUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}


/**
  * @author hnbian
  * @ Description 梯度提升树代码示例
  * @ Date 2019/1/11 10:30
  **/
object GBTClassifier extends App {

  val spark = SparkUtils.getSparkSession("GBTClassifier", 4)

  val filePath = FileUtils.getFilePath("sample_libsvm_data.txt")

  //加载数据
  val data = spark.read.format("libsvm").load(filePath)
  //展示加载出的数据
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
    * |  1.0|(692,[158,159,160...|
    * +-----+--------------------+
    */

  //定义GBTClassifier，注意在Spark中输出（预测列）都有默认的设置，可以不自己设置
  val gbtClassifier = new GBTClassifier()
    .setLabelCol("indexedLabel")//输入label
    .setFeaturesCol("indexedFeatures")
    .setMaxIter(3) //最大迭代次数（numIteration）
    .setStepSize(0.5) // 设置学习率（learningRate）
    .setImpurity("entropy") //计算信息增益的准则 or "gini"
    .setLossType("logistic") // 损失函数的类型(loss)

  // 索引标签，将元数据添加到标签列。
  // Fit on 整个数据集包含索引中的所有标签。
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

  // 将索引标签转换回原始标签。
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and GBT in a Pipeline.链式索引器和管道中的GBT。
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, gbtClassifier, labelConverter))

  // 训练模型。 这也运行索引。
  val model = pipeline.fit(trainingData)

  // 进行预测
  val predictions = model.transform(testData)
  predictions.show(3)
  /**
    * +-----+--------------------+------------+--------------------+--------------------+--------------------+----------+--------------+
    * |label|            features|indexedLabel|     indexedFeatures|       rawPrediction|         probability|prediction|predictedLabel|
    * +-----+--------------------+------------+--------------------+--------------------+--------------------+----------+--------------+
    * |  0.0|(692,[98,99,100,1...|         1.0|(692,[98,99,100,1...|[-1.9064626927749...|[0.02160633849263...|       1.0|           0.0|
    * |  0.0|(692,[100,101,102...|         1.0|(692,[100,101,102...|[-0.7639208415359...|[0.17830969280890...|       1.0|           0.0|
    * |  0.0|(692,[121,122,123...|         1.0|(692,[121,122,123...|[-1.9064626927749...|[0.02160633849263...|       1.0|           0.0|
    * +-----+--------------------+------------+--------------------+--------------------+--------------------+----------+--------------+
    */

  //模型评估
  val (accuracy, precision, recall, f1) = Evaluations.multiClassEvaluate(predictions)
  println("\n\n========= 评估结果 ==========")
  println(s"\n准确率：$accuracy")
  println(s"加权精确率：$precision")
  println(s"加权召回率：$recall")
  println(s"F1值：$f1")
  /**
    * ========= 评估结果=========
    *
    * 准确率：0.9428571428571428
    * 加权精确率：0.9488721804511278
    * 加权召回率：0.9428571428571428
    * F1值：0.9427637721755369
    */

  val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
  println("Learned classification GBT model:\n" + gbtModel.toDebugString)
  /**
    * Learned classification GBT model:
    * GBTClassificationModel (uid=gbtc_38531a8df50a) with 3 trees
    * Tree 0 (weight 1.0):
    * If (feature 406 <= 126.5)
    * If (feature 99 in {2.0})
    * Predict: -1.0
    * Else (feature 99 not in {2.0})
    * Predict: 1.0
    * Else (feature 406 > 126.5)
    * Predict: -1.0
    * Tree 1 (weight 0.5):
    * If (feature 434 <= 79.5)
    * If (feature 184 <= 253.5)
    * Predict: 0.47681168808847024
    * Else (feature 184 > 253.5)
    * Predict: -0.4768116880884694
    * Else (feature 434 > 79.5)
    * If (feature 351 <= 70.5)
    * Predict: -0.4768116880884702
    * Else (feature 351 > 70.5)
    * Predict: -0.47681168808847035
    * Tree 2 (weight 0.5):
    * If (feature 490 <= 27.5)
    * If (feature 99 in {2.0})
    * Predict: -0.3099993569763609
    * Else (feature 99 not in {2.0})
    * If (feature 100 <= 126.5)
    * Predict: 0.3099993569763608
    * Else (feature 100 > 126.5)
    * Predict: 0.30999935697636083
    * Else (feature 490 > 27.5)
    * If (feature 124 <= 188.5)
    * Predict: -0.3099993569763608
    * Else (feature 124 > 188.5)
    * Predict: -0.30999935697636083
    */
}
