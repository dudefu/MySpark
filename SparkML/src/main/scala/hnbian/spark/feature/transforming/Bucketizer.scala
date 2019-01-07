package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.Bucketizer

/**
  * @author hnbian 离散化重组，分箱操作
  *         @ Description
  *         @ Date 2018/12/28 17:11
  **/
object Bucketizer extends App {
  val spark = SparkUtils.getSparkSession("Bucketizer", 4)
  //定义数据集
  val df = spark.createDataFrame(Seq(
    (0, 10),
    (1, 15),
    (2, 30),
    (3, 20),
    (4, 51),
    (5, 60),
    (6, 18)
  )).toDF("id", "age")
  df.show(false)
  /**
    * +---+---+
    * |id |age|
    * +---+---+
    * |0  |10 |
    * |1  |15 |
    * |2  |30 |
    * |3  |20 |
    * |4  |51 |
    * |5  |60 |
    * |6  |18 |
    * +---+---+
    */

  val splits = Array(0,18,35,55,Double.PositiveInfinity)

  //定义转换器 并设置splits
  val bucketizer  = new Bucketizer()
    .setInputCol("age")
    .setOutputCol("age_stage")
    .setSplits(splits)

  bucketizer.transform(df).show(false)
  /**
    * +---+---+---------+
    * |0  |10 |0.0      |
    * |1  |15 |0.0      |
    * |2  |30 |1.0      |
    * |3  |20 |1.0      |
    * |4  |51 |2.0      |
    * |5  |60 |3.0      |
    * |6  |18 |1.0      |
    * +---+---+---------+
    * 18在分类数值上被算在后一个分类中
    */

}
