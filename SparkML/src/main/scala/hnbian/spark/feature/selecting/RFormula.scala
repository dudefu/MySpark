package hnbian.spark.ml.feature.selecting

import utils.SparkUtils
import org.apache.spark.ml.feature.RFormula

/**
  * @author hnbian
  *         @ Description R公式
  *         @ Date 2018/12/29 11:12
  **/
object RFormula extends App {
  val spark = SparkUtils.getSparkSession("RFormula", 4)

  // 定义一个dataFrame
  val dataDF = spark.createDataFrame(Seq(
    (7, "US", 18, 1.0),
    (8, "CA", 12, 0.0),
    (9, "NZ", 15, 0.0)
  )).toDF("id", "country", "hour", "clicked")

  //展示数据
  dataDF.show()

  /**
    * +---+-------+----+-------+
    * | id|country|hour|clicked|
    * +---+-------+----+-------+
    * |  7|     US|  18|    1.0|
    * |  8|     CA|  12|    0.0|
    * |  9|     NZ|  15|    0.0|
    * +---+-------+----+-------+
    */

    //定义评估器 并设置输入输出与计算公式
  val formula = new RFormula()
    .setFormula("clicked ~ country + hour")
    .setFeaturesCol("features")
    .setLabelCol("label")
  //使用评估器训练模型
  val outputModel = formula.fit(dataDF)
  //使用转换器 对测试数据集进行转换 并查看转换后的数据
  outputModel.transform(dataDF).show
  /**
    * +---+-------+----+-------+--------------+-----+
    * | id|country|hour|clicked|      features|label|
    * +---+-------+----+-------+--------------+-----+
    * |  7|     US|  18|    1.0|[0.0,0.0,18.0]|  1.0|
    * |  8|     CA|  12|    0.0|[1.0,0.0,12.0]|  0.0|
    * |  9|     NZ|  15|    0.0|[0.0,1.0,15.0]|  0.0|
    * +---+-------+----+-------+--------------+-----+
    */


}
