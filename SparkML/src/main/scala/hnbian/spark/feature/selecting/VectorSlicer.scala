package hnbian.spark.ml.feature.selecting

import utils.SparkUtils
import java.util.Arrays

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType


/**
  * @author hnbian
  *         @ Description 特征选择 向量机代码示例 【转换器】
  *         @ Date 2018/12/29 9:52
  **/
object VectorSlicer extends App {
  val spark = SparkUtils.getSparkSession("VectorSlicer", 4)


  //定义数据集
  val data = Arrays.asList(Row(Vectors.dense(0.0, 10.0, 0.5)))

  val defaultAttr = NumericAttribute.defaultAttr
  val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)

  println(attrs.toBuffer)
  //ArrayBuffer({"type":"numeric","name":"f1"}, {"type":"numeric","name":"f2"}, {"type":"numeric","name":"f3"})

  val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

  val dataDF = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))

  dataDF.show(false)
  /**
    * +--------------+
    * |userFeatures  |
    * +--------------+
    * |[0.0,10.0,0.5]|
    * +--------------+
    */

  //定义通过下表选择特征的转换器设置输入输出与选择列的下标
  val slicer = new VectorSlicer()
    .setInputCol("userFeatures")
    .setOutputCol("features")
    .setIndices(Array(1, 2)) //使用索引选择特征 or slicer.setIndices(Array(1, 2))

  //使用转换器对数据集进行转换并展示结果
  val output = slicer.transform(dataDF).show()

  /**
    * +--------------+----------+
    * |userFeatures  |features  |
    * +--------------+----------+
    * |[0.0,10.0,0.5]|[10.0,0.5]|
    * +--------------+----------+
    */

  //定义通过列名选择特征的转换器，并设置输入输出与选择列名
  val slicer2 = new VectorSlicer()
    .setInputCol("userFeatures")
    .setOutputCol("features")
    .setNames(Array("f1", "f2"))
  //使用名称选择特征 or slicer.setNames(Array("f1", "f2"))
  //使用转换器对数据进行转换并展示结果
  val output2 = slicer2.transform(dataDF)
  output2.show(false)
  /**
    * +--------------+----------+
    * |userFeatures  |features  |
    * +--------------+----------+
    * |[0.0,10.0,0.5]|[0.0,10.0]|
    * +--------------+----------+
    */

}
