package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

/**
  * @author hnbian
  *         @ Description 向量汇编
  *         @ Date 2018/12/28 18:13
  **/
object VectorAssembler extends App {
  val spark = SparkUtils.getSparkSession("VectorAssembler", 4)

  //定义数据集
  val dataset = spark.createDataFrame(
    Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
  ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
  //展示数据集
  dataset.show()
  /**
    * +---+----+------+--------------+-------+
    * | id|hour|mobile|  userFeatures|clicked|
    * +---+----+------+--------------+-------+
    * |  0|  18|   1.0|[0.0,10.0,0.5]|    1.0|
    * +---+----+------+--------------+-------+
    */

    //定义转换器 并设置要合并的列名
  val assembler = new VectorAssembler()
    .setInputCols(Array("hour", "mobile", "userFeatures")) //设置要合并的列名
    .setOutputCol("features")

  //转换数据集得到含有新向量的数据集
  val output = assembler.transform(dataset)
  println(output.select("features", "clicked").first())
  //[[18.0,1.0,0.0,10.0,0.5],1.0]

  //打印数据集
  output.show(false)
  /**
    * +---+----+------+--------------+-------+-----------------------+
    * |id |hour|mobile|userFeatures  |clicked|features               |
    * +---+----+------+--------------+-------+-----------------------+
    * |0  |18  |1.0   |[0.0,10.0,0.5]|1.0    |[18.0,1.0,0.0,10.0,0.5]|
    * +---+----+------+--------------+-------+-----------------------+
    */


}
