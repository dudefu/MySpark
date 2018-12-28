package hnbian.spark.ml.feature.transforming

import hnbian.spark.SparkUtils
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}

/**
  * @author hnbian
  *         @ Description 
  *         @ Date 2018/12/28 9:59
  **/
object StringIndexer extends App {
  val spark = SparkUtils.getSparkSession("StringIndexer", 4)
  //定义数据集
  val df1 = spark.createDataFrame(Seq(
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (5, "c"))).toDF("id", "category")
  //查看刚刚定义的数据集
  df1.show(false)
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
  //创建一个StringIndexer对象
  val indexer = new StringIndexer()
    .setInputCol("category") //设定输入列名
    .setOutputCol("categoryIndex") //设定输出列名

  //对这个DataFrame进行训练，产生StringIndexerModel对象（转换器）
  val stringIndexerModel = indexer.fit(df1)
  //使用训练出来的模型对数据集进行转换
  val indexed = stringIndexerModel.transform(df1)
  //查看转换后的数据集
  indexed.show(false)
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

  /**
    * 从上看转换后的数据集我们可以看到，
    * StringIndexerModel依次按照出现频率的高低，把字符标签进行了排序，
    * 即出现最多的“a”（3次）被编号成0，“c”（2次）为1，出现最少的“b”（1次）为0。
    */

  /**
    * 如果我们使用StringIndexerModel去转换一个模型内没有出现过的包含“d”标签的DataFrame 会有什么效果？
    * 实际上，如果直接转换的话，Spark会抛出异常，报出“Unseen label: d”的错误。
    * 为了处理这种情况，在模型训练后，可以通过设置setHandleInvalid("skip")来忽略掉那些未出现的标签，
    * 这样，带有未出现标签的行将直接被过滤掉，所下所示：
    */

  //构建一个包含“d”标签的数据集
  val df2 = spark.createDataFrame(Seq(
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (4, "a"),
    (5, "d"))).toDF("id", "category")

  df2.show(false)
  /**
    * +---+--------+
    * |id |category|
    * +---+--------+
    * |0  |a       |
    * |1  |b       |
    * |2  |c       |
    * |3  |a       |
    * |4  |a       |
    * |5  |d       |
    * +---+--------+
    */
    //使用模型转换包含“d”标签的数据集
  val indexed2 = stringIndexerModel.transform(df2)
  //展示转换后的数据集
  //indexed2.show(false)
  //报错：Caused by: org.apache.spark.SparkException: Unseen label: d.  To handle unseen labels, set Param handleInvalid to keep.

  //需要先跳过之前模型训练时不包含的标签再进行转换
  val indexed3 = stringIndexerModel.setHandleInvalid("skip").transform(df2)
  indexed3.show(false)
  /**
    * +---+--------+-------------+
    * |id |category|categoryIndex|
    * +---+--------+-------------+
    * |0  |a       |0.0          |
    * |1  |b       |2.0          |
    * |2  |c       |1.0          |
    * |3  |a       |0.0          |
    * |4  |a       |0.0          |
    * +---+--------+-------------+
    */
  /**
    * 我们可以看到 在这次转换中c的次数已经跟b是相等的都只有一次但是c的索引值仍然是1，因为在模型训练时已经对c定义索引为1。
    */
}
