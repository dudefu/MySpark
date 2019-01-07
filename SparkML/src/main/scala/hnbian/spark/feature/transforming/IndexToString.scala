package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}


/**
  * @author hnbian
  *         @ Description 
  *         @ Date 2018/12/28 10:49
  **/
object IndexToString extends App {

  val spark = SparkUtils.getSparkSession("IndexToString", 4)
  //定义数据集
  val df = spark.createDataFrame(Seq(
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")
  )).toDF("id", "category")
  //展示数据集
  df.show(false)
  /**
    * +---+--------+
    * |id |category|
    * +---+--------+
    * |0  |a       |
    * |1  |b       |
    * |2  |c       |
    * |3  |a       |
    * |4  |a       |
    * |5  |c       |
    * +---+--------+
    */
  //创建StringIndexer 评估器
  val stringIndexer = new StringIndexer()
    .setInputCol("category") //设置输入列
    .setOutputCol("categoryIndex")
  //设置输出列
  //使用刚刚定义的StringIndexer评估器训练数据获得模型
  val indexer = stringIndexer.fit(df)
  //使用模型转换数据
  val df2 = indexer.transform(df)
  df2.show(false)
  /**
    * +---+--------+-------------+
    * |id |category|categoryIndex|
    * +---+--------+-------------+
    * |0  |a       |0.0          |
    * |1  |b       |2.0          |
    * |2  |c       |1.0          |
    * |3  |a       |0.0          |
    * |4  |a       |0.0          |
    * |5  |c       |1.0          |
    * +---+--------+-------------+
    */
  //定义一个IndexToString 转换器
  val converter = new IndexToString()
    .setInputCol("categoryIndex") //设置读取“categoryIndex”上的标签索引
    .setOutputCol("originalCategory") //设置输出到“originalCategory”列
  //转换数据集字符型标签，然后再输出到“originalCategory”列上
  val converted = converter.transform(df2)
  //通过输出“originalCategory”列，可以看到数据集中原有的字符标签
  converted.show(false)
  /**
    * +---+--------+-------------+----------------+
    * |id |category|categoryIndex|originalCategory|
    * +---+--------+-------------+----------------+
    * |0  |a       |0.0          |a               |
    * |1  |b       |2.0          |b               |
    * |2  |c       |1.0          |c               |
    * |3  |a       |0.0          |a               |
    * |4  |a       |0.0          |a               |
    * |5  |c       |1.0          |c               |
    * +---+--------+-------------+----------------+
    */
}
