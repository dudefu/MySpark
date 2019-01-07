package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors


/**
  * @author hnbian
  *         @ Description 
  *         @ Date 2018/12/27 17:40
  **/
object PCA extends App {
  val spark = SparkUtils.getSparkSession("PCA", 4)

  val data = Array(
    Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
  )

  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
  df.show(false)
  /**
    * +---------------------+
    * |features             |
    * +---------------------+
    * |(5,[1,3],[1.0,7.0])  |
    * |[2.0,0.0,3.0,4.0,5.0]|
    * |[4.0,0.0,0.0,6.0,7.0]|
    * +---------------------+
    */
   val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(3) //转换为3维主成分向量
    .fit(df)

  pca.transform(df).show(false)

  /**
    * +---------------------+-----------------------------------------------------------+
    * |features             |pcaFeatures                                                |
    * +---------------------+-----------------------------------------------------------+
    * |(5,[1,3],[1.0,7.0])  |[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
    * |[2.0,0.0,3.0,4.0,5.0]|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
    * |[4.0,0.0,0.0,6.0,7.0]|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
    * +---------------------+-----------------------------------------------------------+
    */
}
