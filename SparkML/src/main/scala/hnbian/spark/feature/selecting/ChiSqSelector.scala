package hnbian.spark.ml.feature.selecting

import utils.SparkUtils
import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel}
import org.apache.spark.ml.linalg.Vectors

/**
  * @author hnbian
  *         @ Description  卡方选择
  *         @ Date 2018/12/29 11:26
  **/
object ChiSqSelector extends App {
  val spark = SparkUtils.getSparkSession("ChiSqSelector", 4)
  //定义数据集 有三个样本，四个特征维度的数据集，标签有1，0两种
  val df = spark.createDataFrame(Seq(
    (1, Vectors.dense(0.0, 0.0, 18.0, 1.0,0.0), 1),
    (2, Vectors.dense(0.0, 1.0, 12.0, 0.0,0.0), 0),
    (3, Vectors.dense(1.0, 0.0, 15.0, 0.1,0.0), 0)
  )).toDF("id", "features", "label")
  //展示数据集
  df.show()
  /**
    * +---+------------------+-----+
    * | id|          features|label|
    * +---+------------------+-----+
    * |  1|[0.0,0.0,18.0,1.0]|    1|
    * |  2|[0.0,1.0,12.0,0.0]|    0|
    * |  3|[1.0,0.0,15.0,0.1]|    0|
    * +---+------------------+-----+
    */

  /**
    * 卡方选择进行特征选择器的训练，
    * 为了观察地更明显，我们设置只选择和标签关联性最强的一个特征
    * （可以通过setNumTopFeatures()方法进行设置）
    */

  val selector = new ChiSqSelector()
    .setNumTopFeatures(2) // 选取与标签关联性最强的几个特征 默认全选
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setOutputCol("selectedFeature")

  //训练模型
  val selectorModel = selector.fit(df)

  //使用模型对数据集进行转换并展示数据
  selectorModel.transform(df).show(false)

  /**
    * .setNumTopFeatures(1)
    * +---+------------------+-----+---------------+
    * | id|          features|label|selectedFeature|
    * +---+------------------+-----+---------------+
    * |  1|[0.0,0.0,18.0,1.0]|    1|         [18.0]|
    * |  2|[0.0,1.0,12.0,0.0]|    0|         [12.0]|
    * |  3|[1.0,0.0,15.0,0.1]|    0|         [15.0]|
    * +---+------------------+-----+---------------+
    */
  /**
    * .setNumTopFeatures(2)
    * +---+----------------------+-----+---------------+
    * |id |features              |label|selectedFeature|
    * +---+----------------------+-----+---------------+
    * |1  |[0.0,0.0,18.0,1.0,0.0]|1    |[18.0,1.0]     |
    * |2  |[0.0,1.0,12.0,0.0,0.0]|0    |[12.0,0.0]     |
    * |3  |[1.0,0.0,15.0,0.1,0.0]|0    |[15.0,0.1]     |
    * +---+----------------------+-----+---------------+
    */
}
