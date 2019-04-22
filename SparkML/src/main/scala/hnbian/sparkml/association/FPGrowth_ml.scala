package hnbian.sparkml.association

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.fpm.{FPGrowth, FPGrowthModel}
import utils.FileUtils

/**
  * @author hnbian 2019/4/17 15:18
  *         spark 关联规则，基于DataFrame版本
  */
object FPGrowth_ml extends App {
  val spark = SparkUtils.getSparkSession("FPGrowth", 4)
  import spark.implicits._

  //最小支持度
  val minSupport = 0.055
  //最小置信度
  val minConfidence = 0.2
  //数据分区
  val numPartitions = 10
  //取出数据
  val filePath = FileUtils.getFilePath("association_rules.txt")
  println(filePath)
  val dataDF = spark.sparkContext.textFile(filePath).map(x => x.split(",")).toDF("items")

  val fpg = new FPGrowth()
  val model:FPGrowthModel = fpg.setItemsCol("items")
    .setMinConfidence(minConfidence)
    .setMinSupport(minSupport)
    .fit(dataDF)
  //筛选出三项数据
  model.freqItemsets.filter("size(items) =3").show(5)
  /**
    * +---------------+----+
    * |          items|freq|
    * +---------------+----+
    * |[C 2, A 2, D 2]|  53|
    * |[C 2, D 2, B 4]|  52|
    * |[C 2, D 2, H 4]|  57|
    * |[C 2, E 1, A 2]|  52|
    * |[B 4, C 4, H 4]|  54|
    * +---------------+----+
    */
  model.associationRules.filter("size(antecedent)>=3").show(5)
  /**
    * +---------------+----------+------------------+------------------+
    * |     antecedent|consequent|        confidence|              lift|
    * +---------------+----------+------------------+------------------+
    * |[E 3, D 2, B 4]|     [H 4]|0.6419753086419753|1.4386434627398483|
    * |[A 2, D 2, H 4]|     [F 4]| 0.574468085106383| 2.016057808109193|
    * |[F 4, A 2, D 2]|     [H 4]|0.7297297297297297|1.6352979485509607|
    * |[F 4, D 2, H 4]|     [A 2]|0.7397260273972602|1.9378738182519775|
    * |[D 2, B 4, H 4]|     [E 3]|0.5842696629213483|2.2269294529379255|
    * +---------------+----------+------------------+------------------+
    */
}
