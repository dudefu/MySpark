package hnbian.spark.ml.feature.transforming
import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors


object PolynomialExpansion extends App {
  val spark = SparkUtils.getSparkSession("PolynomialExpansion",4)


  //定义一个数 每行两个元素
  val data = Array(
    Vectors.dense(-2.0, 2.3),
    Vectors.dense(0.0, 0.0),
    Vectors.dense(0.6, -1.1)
  )
  //创建数据集
  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
  //打印数据集
  df.show(false)
  /**
    * +----------+
    * |features  |
    * +----------+
    * |[-2.0,2.3]|
    * |[0.0,0.0] |
    * |[0.6,-1.1]|
    * +----------+
    */
  //定义一个多项式展开（升维）转换器，
  val polynomialExpansion = new PolynomialExpansion()
    .setInputCol("features")
    .setOutputCol("polyFeatures")
    .setDegree(3) //扩展到3维向量

  //调用转换器transform()方法对数据集进行转换
  val polyDF = polynomialExpansion.transform(df)
  //查看转换后的结果
  polyDF.show(false)

  /**
    * +----------+--------------------------------------------------------------------------------------------+
    * |features  |polyFeatures                                                                                |
    * +----------+--------------------------------------------------------------------------------------------+
    * |[-2.0,2.3]|[-2.0,4.0,-8.0,2.3,-4.6,9.2,5.289999999999999,-10.579999999999998,12.166999999999996]       |
    * |[0.0,0.0] |[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]                                                       |
    * |[0.6,-1.1]|[0.6,0.36,0.216,-1.1,-0.66,-0.396,1.2100000000000002,0.7260000000000001,-1.3310000000000004]|
    * +----------+--------------------------------------------------------------------------------------------+
    */


}
