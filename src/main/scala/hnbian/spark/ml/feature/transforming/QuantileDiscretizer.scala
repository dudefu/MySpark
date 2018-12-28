package hnbian.spark.ml.feature.transforming

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.feature.QuantileDiscretizer

/**
  * @author hnbian
  *         @ Description  分数求解器（与分箱一样对连续数据做离散化）
  *         @ Date 2018/12/28 17:31
  **/
object QuantileDiscretizer extends App {
  val spark = SparkUtils.getSparkSession("QuantileDiscretizer", 4)
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

    //定义评估器 设置相关参数
  val quantileDiscretizer = new QuantileDiscretizer()
    .setInputCol("age")
    .setOutputCol("age_stage")
    .setNumBuckets(4) //将数据分为4类
    .setRelativeError(0.1) //精度设置为0.1

  //训练模型
  val model = quantileDiscretizer.fit(df)
  //使用模型转换数据并展示数据
  model.transform(df).show(false)
  /**
    * +---+---+---------+
    * |id |age|age_stage|
    * +---+---+---------+
    * |0  |10 |0.0      |
    * |1  |15 |1.0      |
    * |2  |30 |2.0      |
    * |3  |20 |2.0      |
    * |4  |51 |3.0      |
    * |5  |60 |3.0      |
    * |6  |18 |1.0      |
    * +---+---+---------+
    */

}
