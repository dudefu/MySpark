package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors


/**
  * @author hnbian
  *         @ Description 元素乘积
  *         @ Date 2018/12/28 17:49
  **/
object ElemtwiseProduct extends App {
  val spark = SparkUtils.getSparkSession("ElemtwiseProduct", 4)
  // 创建一些向量数据; 也适用于稀疏向量
  val dataFrame = spark.createDataFrame(Seq(
    ("a", Vectors.dense(1.0, 2.0, 3.0)),
    ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")
  dataFrame.show()
  /**
    * +---+-------------+
    * | id|       vector|
    * +---+-------------+
    * |  a|[1.0,2.0,3.0]|
    * |  b|[4.0,5.0,6.0]|
    * +---+-------------+
    */

    //定义转换向量
  val transformingVector = Vectors.dense(0.0, 1.0, 2.0)

  println(s"transformingVector = ${transformingVector}")
  //transformingVector = [0.0,1.0,2.0]

  //定义一个转换器
  val transformer = new ElementwiseProduct()
    .setScalingVec(transformingVector)
    .setInputCol("vector")
    .setOutputCol("transformedVector")

  // 使用转换器对数据集进行转换
  transformer.transform(dataFrame).show()
  /**
    * 矩阵中的每个元素分别与转换向量中的元素相乘
    * +---+-------------+-----------------+
    * | id|       vector|transformedVector|
    * +---+-------------+-----------------+
    * |  a|[1.0,2.0,3.0]|    [0.0,2.0,6.0]|
    * |  b|[4.0,5.0,6.0]|   [0.0,5.0,12.0]|
    * +---+-------------+-----------------+
    */


}
