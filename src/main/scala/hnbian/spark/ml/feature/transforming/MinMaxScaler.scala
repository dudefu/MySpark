package hnbian.spark.ml.feature.transforming

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors

/**
  * @author hnbian 特征转换归一化代码示例
  *         @ Description
  *         @ Date 2018/12/28 16:34
  **/
object MinMaxScaler extends App {
  val spark = SparkUtils.getSparkSession("MinMaxScaler", 4)

  //定义数据集
  val dataFrame = spark.createDataFrame(Seq(
    (0, Vectors.dense(1.0, 0.5, 3.0)),
    (1, Vectors.dense(2.0, 11.0, 1.0)),
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

  //定义MinMaxScaler评估器 设置输入输出列
  val scaler = new MinMaxScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  // 训练测试数据集并生成一个model
  val scalerModel = scaler.fit(dataFrame)

  // 使用model 对测试数据集进行转换
  val scaledData = scalerModel.transform(dataFrame)

  scaledData.show(false)
  // 每维特征线性地映射，最小值映射到0，最大值映射到1。
  /**
    * +---+--------------+----------------------------+
    * |id |features      |scaledFeatures              |
    * +---+--------------+----------------------------+
    * |0  |[1.0,0.5,3.0] |[0.0,0.0,1.0]               |
    * |1  |[2.0,11.0,1.0]|[0.3333333333333333,1.0,0.0]|
    * |2  |[4.0,10.0,2.0]|[1.0,0.9047619047619048,0.5]|
    * +---+--------------+----------------------------+
    */

}
