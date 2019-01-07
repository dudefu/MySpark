package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.Binarizer

/**
  * @author hnbian
  *         @ Description 
  *         @ Date 2018/12/27 16:39
  **/
object Binarizer extends App {
  val spark = SparkUtils.getSparkSession("Binarizer", 4)

  val dataDF = spark
    .createDataFrame(Array((0, 0.1), (1, 0.8), (2, 0.2), (3, 0.6), (4, 0.5)))
    .toDF("label", "feature")
  dataDF.show(false)
  /**
    * +-----+-------+
    * |label|feature|
    * +-----+-------+
    * |0    |0.1    |
    * |1    |0.8    |
    * |2    |0.2    |
    * |3    |0.6    |
    * |4    |0.5    |
    * +-----+-------+
    */

  val binarizer: Binarizer = new Binarizer()
    .setInputCol("feature")
    .setOutputCol("binarized_feature")
    .setThreshold(0.5) // 设置阈值 if( value > 0.5 ) 1.0 else 0.0


  val binarizedDF = binarizer.transform(dataDF)
  binarizedDF.show(false)
  /**
    * +-----+-------+-----------------+
    * |label|feature|binarized_feature|
    * +-----+-------+-----------------+
    * |0    |0.1    |0.0              |
    * |1    |0.8    |1.0              |
    * |2    |0.2    |0.0              |
    * |3    |0.6    |1.0              |
    * |4    |0.5    |0.0              |
    * +-----+-------+-----------------+
    */
  val binarizedFeatures = binarizedDF.select("binarized_feature")
  binarizedFeatures.collect().foreach(println)
  /**
    * [0.0]
    * [1.0]
    * [0.0]
    * [1.0]
    * [0.0]
    */
  binarizedFeatures.show(false)
  /**
    * +-----------------+
    * |binarized_feature|
    * +-----------------+
    * |0.0              |
    * |1.0              |
    * |0.0              |
    * |1.0              |
    * |0.0              |
    * +-----------------+
    */
}
