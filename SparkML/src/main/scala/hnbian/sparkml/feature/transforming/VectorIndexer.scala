package hnbian.spark.ml.feature.transforming

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * @author hnbian
  *         @ Description
  *         @ Date 2018/12/28 12:09
  **/
object VectorIndexer extends App {
  val spark = SparkUtils.getSparkSession("VectorIndexer", 4)

  val data = Seq(
    Vectors.dense(-1.0, 1.0, 1.0),
    Vectors.dense(-1.0, 3.0, 1.0),
    Vectors.dense(0.0, 5.0, 1.0))
  //定义数据集
  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
  //展示数据集
  df.show()
  /**
    * +--------------+
    * |      features|
    * +--------------+
    * |[-1.0,1.0,1.0]|
    * |[-1.0,3.0,1.0]|
    * | [0.0,5.0,1.0]|
    * +--------------+
    */
  //定义 VectorIndexer 评估器

  val indexer = new VectorIndexer().
    setInputCol("features").
    setOutputCol("indexed").
    setMaxCategories(2) //设置最大种类别为2，即每列种的元素类型不能大于2种，比如中间列为1，3，5 为3种

  //使用VectorIndexer训练出模型，来决定哪些特征需要被作为类别特征
  val indexerModel = indexer.fit(df)
  /**
    * 即只有种类小于2的特征才被认为是类别型特征，否则被认为是连续型特征
    * 可以通过VectorIndexerModel的categoryMaps成员来获得被转换的特征及其映射，
    * 这里可以看到共有两个特征被转换，分别是下标为0和2的元素。
    * 可以看到，0号特征只有-1，0两种取值，分别被映射成0，1，而2号特征只有1种取值，被映射成0。
    */
  indexerModel.transform(df).show()
  /**
    * +--------------+-------------+
    * |      features|      indexed|
    * +--------------+-------------+
    * |[-1.0,1.0,1.0]|[1.0,1.0,0.0]|
    * |[-1.0,3.0,1.0]|[1.0,3.0,0.0]|
    * | [0.0,5.0,1.0]|[0.0,5.0,0.0]|
    * +--------------+-------------+
    */
  val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
  println(s"Chose ${categoricalFeatures.size} categorical features: " + categoricalFeatures.mkString(", "))
  //Chose 2 categorical features: 0, 2

}
