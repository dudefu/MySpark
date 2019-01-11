package hnbian.sparkml.algorithms.classification

import hnbian.spark.utils.SparkUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row
import utils.FileUtils
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

/**
  * @author hnbian
  *         @ Description
  *         @ Date 2019/1/11 14:52
  **/
object MultilayerPerceptronClassifier extends App {
  val spark = SparkUtils.getSparkSession("MultilayerPerceptronClassifier", 4)
  val filePath = FileUtils.getFilePath("sample_multiclass_classification_data.txt")
  //导入测试数据
  val data = spark.read.format("libsvm").load(filePath)
  //打印数据
  data.show()
  /**
    * +-----+--------------------+
    * |label|            features|
    * +-----+--------------------+
    * |  1.0|(4,[0,1,2,3],[-0....|
    * |  1.0|(4,[0,1,2,3],[-0....|
    * |  1.0|(4,[0,1,2,3],[-0....|
    * |  1.0|(4,[0,1,2,3],[-0....|
    * |  0.0|(4,[0,1,2,3],[0.1...|
    * +-----+--------------------+
    */

  // 将数据拆分为训练和测试
  val Array(train, test) = data.randomSplit(Array(0.6, 0.4), seed = 1234L)

  /**
    * 为神经网络指定图层：
    * 输入特征有4个维度
    * 隐层有两个中间层 分别有5、4个神经元
    * 结果有3类输出
    */
  val layers = Array[Int](4, 5, 4, 3)
  // 创建训练器并设置其参数
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)
  // train the model
  val model = trainer.fit(train)
  // 计算测试集的准确性
  val predictions = model.transform(test)
  predictions.show(3)
  /**
    * +-----+--------------------+--------------------+--------------------+----------+
    * |label|            features|       rawPrediction|         probability|prediction|
    * +-----+--------------------+--------------------+--------------------+----------+
    * |  0.0|(4,[0,1,2,3],[-0....|[-29.588369001638...|[2.63020383878084...|       2.0|
    * |  0.0|(4,[0,1,2,3],[-0....|[125.657894478296...|[1.0,1.4484875476...|       0.0|
    * |  0.0|(4,[0,1,2,3],[-0....|[126.190155254739...|[1.0,5.1578089761...|       0.0|
    * +-----+--------------------+--------------------+--------------------+----------+
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
    * 准确率：0.9019607843137255
    * 加权精确率：0.9111111111111112
    * 加权召回率：0.9019607843137256
    * F1值：0.9019607843137256
    */
}
