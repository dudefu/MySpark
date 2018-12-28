package hnbian.spark.ml.feature.transforming

import hnbian.spark.SparkUtils
import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg.Vectors


/**
  * @author hnbian
  *         @ Description 离散余弦变换
  *         @ Date 2018/12/28 9:26
  **/
object DCT extends App {
  val spark = SparkUtils.getSparkSession("DCT", 4)

  val data = Seq(
    Vectors.dense(0.0, 1.0, -2.0, 3.0),
    Vectors.dense(-1.0, 2.0, 4.0, -7.0),
    Vectors.dense(14.0, -2.0, -5.0, 1.0))
  //定义数据集
  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
  //打印定义的数据集
  df.show(false)
  /**
    * +--------------------+
    * |features            |
    * +--------------------+
    * |[0.0,1.0,-2.0,3.0]  |
    * |[-1.0,2.0,4.0,-7.0] |
    * |[14.0,-2.0,-5.0,1.0]|
    * +--------------------+
    */
  val dct = new DCT()
    .setInputCol("features")
    .setOutputCol("featuresDCT")
    .setInverse(false)

  val dctDF = dct.transform(df)
  dctDF.show(false)
  /**
    * +--------------------+----------------------------------------------------------------+
    * |features            |featuresDCT                                                     |
    * +--------------------+----------------------------------------------------------------+
    * |[0.0,1.0,-2.0,3.0]  |[1.0,-1.1480502970952693,2.0000000000000004,-2.7716385975338604]|
    * |[-1.0,2.0,4.0,-7.0] |[-1.0,3.378492794482933,-7.000000000000001,2.9301512653149677]  |
    * |[14.0,-2.0,-5.0,1.0]|[4.0,9.304453421915744,11.000000000000002,1.5579302036357163]   |
    * +--------------------+----------------------------------------------------------------+
    */

}
