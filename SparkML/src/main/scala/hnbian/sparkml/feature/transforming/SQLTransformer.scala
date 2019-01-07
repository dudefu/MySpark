package hnbian.spark.ml.feature.transforming

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.feature.SQLTransformer

/**
  * @author hnbian SQL转换器
  *         @ Description
  *         @ Date 2018/12/28 18:02
  **/
object SQLTransformer extends App {
  val spark = SparkUtils.getSparkSession("SQLTransformer", 4)
  //定义数据集
  val df = spark.createDataFrame(Seq(
    (0, 1.0, 3.0),
    (2, 2.0, 5.0))
  ).toDF("id", "v1", "v2")
  //查看数据集
  df.show(false)
  /**
    * +---+---+---+
    * |id |v1 |v2 |
    * +---+---+---+
    * |0  |1.0|3.0|
    * |2  |2.0|5.0|
    * +---+---+---+
    */
  //定义转换器，并设置转换sql语句
  val sqlTrans = new SQLTransformer()
    .setStatement("SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
  //转换数据并查看结果
  sqlTrans.transform(df).show()

  /**
    * +---+---+---+---+----+
    * | id| v1| v2| v3|  v4|
    * +---+---+---+---+----+
    * |  0|1.0|3.0|4.0| 3.0|
    * |  2|2.0|5.0|7.0|10.0|
    * +---+---+---+---+----+
    */
}
