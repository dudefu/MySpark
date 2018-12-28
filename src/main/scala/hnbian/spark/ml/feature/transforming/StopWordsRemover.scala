package hnbian.spark.ml.feature.transforming

import hnbian.spark.SparkUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author hnbian
  *         @ Description 特征转换 停用词移除代码示例
  *         @ Date 2018/12/27 15:37
  **/
object StopWordsRemover extends App {
  //获取的SparkSession
  val spark = SparkUtils.getSparkSession("StopWordsRemover",4)

  import org.apache.spark.ml.feature.StopWordsRemover

  val remover = new StopWordsRemover()
    .setInputCol("raw")
    .setOutputCol("filtered")
    //.setStopWords(Array("saw"))//指定停用词并且只过滤掉停用词
    //.setCaseSensitive(true) //是否区分大小写，默认false 不区分

  //定义数据集
  val dataDF = spark.createDataFrame(Seq(
    (0, Seq("I", "saw", "the", "red", "baloon")),
    (1, Seq("Mary", "had", "a", "little", "lamb"))
  )).toDF("id", "raw")

  dataDF.show(false)
  //转换数据并查看结果
  val modelDF = remover.transform(dataDF)

  modelDF.show(false)
  /**
    * +---+----------------------------+--------------------+
    * |id |raw                         |filtered            |
    * +---+----------------------------+--------------------+
    * |0  |[I, saw, the, red, baloon]  |[saw, red, baloon]  |
    * |1  |[Mary, had, a, little, lamb]|[Mary, little, lamb]|
    * +---+----------------------------+--------------------+
    */
}
