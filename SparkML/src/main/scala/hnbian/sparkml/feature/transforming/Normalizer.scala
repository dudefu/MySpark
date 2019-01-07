package hnbian.spark.ml.feature.transforming

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.linalg.Vectors

/**
  * @author hnbian
  *         @ Description 正则化
  *         @ Date 2018/12/28 15:09
  **/
object Normalizer extends App {
  val spark = SparkUtils.getSparkSession("Normalizer", 4)

  import org.apache.spark.ml.feature.Normalizer

  //获取文件路径
  //val path = FileUtils.getFilePath("sample_libsvm_data_bak.txt")

  //打印文件路径
  //println(path)

  //根据测试文件创建数据集
  //val dataFrame = spark.read.format("libsvm").load(path)

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
  // 使用 $L^1$ norm  正则化每个向量
  val normalizer = new Normalizer()
    .setInputCol("features")
    .setOutputCol("normFeatures")
    .setP(1.0) //默认2

  // 将每一行的规整为1阶范数为1的向量，1阶范数即所有值绝对值之和。
  val l1NormData = normalizer.transform(dataFrame)
  l1NormData.show(false)
  /**
    * +---+--------------+------------------+
    * |id |features      |normFeatures      |
    * +---+--------------+------------------+
    * |0  |[1.0,0.5,-1.0]|[0.4,0.2,-0.4]    |
    * |1  |[2.0,1.0,1.0] |[0.5,0.25,0.25]   |
    * |2  |[4.0,10.0,2.0]|[0.25,0.625,0.125]|
    * +---+--------------+------------------+
    */

  // 正则化每个向量到无穷阶范数 （无穷范数——向量中最大元素的绝对值）
  val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
  println("Normalized using L^inf norm")
  lInfNormData.show()
  /**
    * +---+--------------+--------------+
    * | id|      features|  normFeatures|
    * +---+--------------+--------------+
    * |  0|[1.0,0.5,-1.0]|[1.0,0.5,-1.0]|
    * |  1| [2.0,1.0,1.0]| [1.0,0.5,0.5]|
    * |  2|[4.0,10.0,2.0]| [0.4,1.0,0.2]|
    * +---+--------------+--------------+
    */
}
