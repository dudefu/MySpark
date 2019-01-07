package hnbian.spark.ml.feature.transforming

import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, StringIndexer}

/**
  * @author hnbian
  *         @ Description 特征转换独热编码示例
  *         @ Date 2018/12/28 11:12
  **/
object OneHotEncoderEstimator extends App {

  import utils.SparkUtils

  val spark = SparkUtils.getSparkSession("OneHotEncoder", 4)

  // 首先创建一个DataFrame，其包含一列类别性特征，需要注意的是，
  // 在使用OneHotEncoder进行转换前，DataFrame需要先使用StringIndexer将原始标签数值化：
  val df = spark.createDataFrame(Seq(
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c"),
    (6, "d"),
    (7, "d"),
    (8, "d"),
    (9, "d"),
    (10, "e"),
    (11, "e"),
    (12, "e"),
    (13, "e"),
    (14, "e")
  )).toDF("id", "category")
  df.show()
  /**
    * +---+--------+
    * | id|category|
    * +---+--------+
    * |  0|       a|
    * |  1|       b|
    * |  2|       c|
    * |  3|       a|
    * |  4|       a|
    * |  5|       c|
    * |  6|       d|
    * |  7|       d|
    * |  8|       d|
    * |  9|       d|
    * | 10|       e|
    * | 11|       e|
    * | 12|       e|
    * | 13|       e|
    * | 14|       e|
    * +---+--------+
    */
  //定义StringIndexer 评估器并训练出模型
  val indexerModel = new StringIndexer().
    setInputCol("category").
    setOutputCol("categoryIndex").
    fit(df)

  //对数据集进行转换
  val indexed = indexerModel.transform(df)
  //查看转换后的数据
  indexed.show(false)
  /**
    * a 3次，b 1次，c 2次，d 4次，e 5次
    * e 5次、d 4次、a 3次、c 2次、b 1次
    * +---+--------+-------------+
    * |id |category|categoryIndex|
    * +---+--------+-------------+
    * |0  |a       |2.0          |
    * |1  |b       |4.0          |
    * |2  |c       |3.0          |
    * |3  |a       |2.0          |
    * |4  |a       |2.0          |
    * |5  |c       |3.0          |
    * |6  |d       |1.0          |
    * |7  |d       |1.0          |
    * |8  |d       |1.0          |
    * |9  |d       |1.0          |
    * |10 |e       |0.0          |
    * |11 |e       |0.0          |
    * |12 |e       |0.0          |
    * |13 |e       |0.0          |
    * |14 |e       |0.0          |
    * +---+--------+-------------+
    */

  /**
    * 创建OneHotEncoder对象对处理后的DataFrame进行编码，
    * 可以看见，编码后的二进制特征呈稀疏向量形式，
    * 与StringIndexer编码的顺序相同，需注意的是最后一个Category（”b”）被编码为全0向量，
    * 若希望”b”也占有一个二进制特征，则可在创建OneHotEncoder时指定setDropLast(false)。
    */
  //创建一个OneHotEncoderEstimator 评估器
  val encoder = new OneHotEncoderEstimator()
    .setInputCols(Array("categoryIndex"))
    .setOutputCols(Array("categoryVec"))
    //.setDropLast(false) //默认为true 是否删除最后的类别，如共有五个类别输入，删除最后一个则剩下四个类别

  //使用评估器训练模型
  val encoderModel =  encoder.fit(indexed)
  //转换数据集并查看转换后的结果集
  encoderModel.transform(indexed).show(false)
  /**
    * a 3次，b 1次，c 2次，d 4次，e 5次
    * e 5次、d 4次、a 3次、c 2次、b 1次
    * +---+--------+-------------+-------------+
    * |id |category|categoryIndex|categoryVec  |
    * +---+--------+-------------+-------------+
    * |0  |a       |2.0          |(4,[2],[1.0])|
    * |1  |b       |4.0          |(4,[],[])    |
    * |2  |c       |3.0          |(4,[3],[1.0])|
    * |3  |a       |2.0          |(4,[2],[1.0])|
    * |4  |a       |2.0          |(4,[2],[1.0])|
    * |5  |c       |3.0          |(4,[3],[1.0])|
    * |6  |d       |1.0          |(4,[1],[1.0])|
    * |7  |d       |1.0          |(4,[1],[1.0])|
    * |8  |d       |1.0          |(4,[1],[1.0])|
    * |9  |d       |1.0          |(4,[1],[1.0])|
    * |10 |e       |0.0          |(4,[0],[1.0])|
    * |11 |e       |0.0          |(4,[0],[1.0])|
    * |12 |e       |0.0          |(4,[0],[1.0])|
    * |13 |e       |0.0          |(4,[0],[1.0])|
    * |14 |e       |0.0          |(4,[0],[1.0])|
    * +---+--------+-------------+-------------+
    */

  //.setDropLast(false) 设置为false 时打印数据
  /**
    * +---+--------+-------------+-------------+
    * |id |category|categoryIndex|categoryVec  |
    * +---+--------+-------------+-------------+
    * |0  |a       |2.0          |(5,[2],[1.0])|
    * |1  |b       |4.0          |(5,[4],[1.0])|
    * |2  |c       |3.0          |(5,[3],[1.0])|
    * |3  |a       |2.0          |(5,[2],[1.0])|
    * |4  |a       |2.0          |(5,[2],[1.0])|
    * |5  |c       |3.0          |(5,[3],[1.0])|
    * |6  |d       |1.0          |(5,[1],[1.0])|
    * |7  |d       |1.0          |(5,[1],[1.0])|
    * |8  |d       |1.0          |(5,[1],[1.0])|
    * |9  |d       |1.0          |(5,[1],[1.0])|
    * |10 |e       |0.0          |(5,[0],[1.0])|
    * |11 |e       |0.0          |(5,[0],[1.0])|
    * |12 |e       |0.0          |(5,[0],[1.0])|
    * |13 |e       |0.0          |(5,[0],[1.0])|
    * |14 |e       |0.0          |(5,[0],[1.0])|
    * +---+--------+-------------+-------------+
    */
}
