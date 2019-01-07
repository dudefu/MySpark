package hnbian.spark.ml.feature.transforming

import hnbian.spark.ml.feature.transforming.Normalizer.spark
import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.linalg.Vectors

/**
  * @author hnbian
  *         @ Description  最大值-平均值缩放
  *         @ Date 2018/12/28 16:54
  **/
object MaxAbsScaler extends App {
  val spark = SparkUtils.getSparkSession("MaxAbsScaler", 4)

  //创建数据集
  val dataFrame = spark.createDataFrame(Seq(
    (0, Vectors.dense(1.0, 0.5, -1.0)),
    (1, Vectors.dense(2.0, 1.0, 1.0)),
    (2, Vectors.dense(4.0, 10.0, 2.0))
  )).toDF("id", "features")

  dataFrame.show( false)
  /**
    * +---+--------------+
    * |id |features      |
    * +---+--------------+
    * |0  |[1.0,0.5,-1.0]|
    * |1  |[2.0,1.0,1.0] |
    * |2  |[4.0,10.0,2.0]|
    * +---+--------------+
    */
    //定义MaxAbsScaler 评估器
  val scaler = new MaxAbsScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  // 使用评估器训练数据得到一个模型
  val scalerModel = scaler.fit(dataFrame)

  // 使用模型对数据进行转换
  val scaledData = scalerModel.transform(dataFrame)
  scaledData.select("features", "scaledFeatures").show()

  // 每一维的绝对值的最大值为[4, 10, 2]
  /**
    * +--------------+----------------+
    * |      features|  scaledFeatures|
    * +--------------+----------------+
    * |[1.0,0.5,-1.0]|[0.25,0.05,-0.5]|
    * | [2.0,1.0,1.0]|   [0.5,0.1,0.5]|
    * |[4.0,10.0,2.0]|   [1.0,1.0,1.0]|
    * +--------------+----------------+
    */
}
