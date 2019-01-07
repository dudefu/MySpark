package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors

/**
  * @author hnbian
  *         @ Description 标准缩放
  *         @ Date 2018/12/28 16:03
  **/
object StandardScaler extends App {
  val spark = SparkUtils.getSparkSession("StandardScaler", 4)
  //定义数据集
  val dataFrame = spark.createDataFrame(Seq(
    (0, Vectors.dense(1.0, 0.5, -1.0)),
    (1, Vectors.dense(2.0, 1.0, 1.0)),
    (2, Vectors.dense(4.0, 10.0, 2.0))
  )).toDF("id", "features")

  dataFrame.show(false)
  /**
    * +---+--------------+
    * |id |features      |
    * +---+--------------+
    * |0  |[1.0,0.5,-1.0]|
    * |1  |[2.0,1.0,1.0] |
    * |2  |[4.0,10.0,2.0]|
    * +---+--------------+
    */
  //定义StandardScaler 评估器 并设置输入输出与相关参数
  val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(true) //是否将数据标准化到单位标准差
    .setWithMean(false) //是否变换为0均值

  //训练数据产生一个模型
  val scalerModel = scaler.fit(dataFrame)

  //使用模型转换数据
  val scaledData = scalerModel.transform(dataFrame)
  //查看转换后的数据集
  scaledData.show(false)
  /**
    * // 将每一列的标准差缩放到1
    * +---+--------------+------------------------------------------------------------+
    * |id |features      |scaledFeatures                                              |
    * +---+--------------+------------------------------------------------------------+
    * |0  |[1.0,0.5,-1.0]|[0.6546536707079771,0.09352195295828246,-0.6546536707079772]|
    * |1  |[2.0,1.0,1.0] |[1.3093073414159542,0.18704390591656492,0.6546536707079772] |
    * |2  |[4.0,10.0,2.0]|[2.6186146828319083,1.8704390591656492,1.3093073414159544]  |
    * +---+--------------+------------------------------------------------------------+
    */
}
