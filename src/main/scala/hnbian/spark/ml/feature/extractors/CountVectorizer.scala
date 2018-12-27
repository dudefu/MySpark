package hnbian.spark.ml.feature.extractors

import hnbian.spark.SparkUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

/**
  * @author hnbian
  *         @ Description
  *         @ Date 2018/12/27 11:27
  **/
object CountVectorizer extends App {
  val spark  = SparkUtils.getSparkSession("CountVectorizer",4)

  val df = spark.createDataFrame(Seq(
    (0, Array("a", "b", "c", "E", "E", "E", "d", "d")),
    (1, Array("a", "b", "b", "E", "c", "a"))
  )).toDF("id", "words")

  df.show(false)
  /**
    * +---+------------------+
    * | id|             words|
    * +---+------------------+
    * |  0|      [a, b, c, d]|
    * |  1|[a, b, b, E, c, a]|
    * +---+------------------+
    *//**
    * 定义一个CountVectorizer，并设定相关超参数
    */
  val cvModel: CountVectorizerModel = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("features")
    .setVocabSize(3) //设定词汇表的最大量为3（最多选取三个词）
    .setMinDF(2) //设定词汇表中的词至少要在2个文档中出现过，以过滤那些偶然出现的词汇
    .fit(df)

  //可以通过CountVectorizerModel的vocabulary成员获得到模型的词汇表：
  println(cvModel.vocabulary.toBuffer)
  //ArrayBuffer(E, b, a)
  //d 虽然出现过两次但是只在一个文档中出现过
  //c 虽然满足设置的条件但是 出现次数只有两次 比其他词都少所以被过滤掉了

  //使用模型对文档进行转换
  cvModel.transform(df).show(false)
  /**
    * +---+------------------------+-------------------------+-------------------------------+
    * |id |words                   |features                 | 按照取词E、b、a的顺序求出现次数 |
    * +---+------------------------+-------------------------+-------------------------------+
    * |0  |[a, b, c, E, E, E, d, d]|(3,[0,1,2],[3.0,1.0,1.0])|  E 出现3次，b 1次，a 1次       |
    * |1  |[a, b, b, E, c, a]      |(3,[0,1,2],[1.0,2.0,2.0])|  E 出现1次，b 2次，a 2次       |
    * +---+------------------------+-------------------------+-------------------------------+
    */

  val cvm = new CountVectorizerModel(Array("a", "b", "c")).
    setInputCol("words").
    setOutputCol("features")

  println(cvm.vocabulary.toBuffer)
  //ArrayBuffer(a, b, c)
  cvm.transform(df).show(false)
  /**
    * +---+------------------------+-------------------------+-------------------------------+
    * |id |words                   |features                 | 按照取词a、b、c的顺序求出现次数 |
    * +---+------------------------+-------------------------+-------------------------------+
    * |0  |[a, b, c, E, E, E, d, d]|(3,[0,1,2],[1.0,1.0,1.0])| a 出现1次，b 1次，c 1次        |
    * |1  |[a, b, b, E, c, a]      |(3,[0,1,2],[2.0,2.0,1.0])| a 出现2次，b 2次，c 1次        |
    * +---+------------------------+-------------------------+--------------------------------+
    */
}
