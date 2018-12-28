package hnbian.spark.ml.feature.transforming

import hnbian.spark.utils.SparkUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}

/**
  * @author hnbian
  *         @ Description 特征转换 分词器代码示例
  *         @ Date 2018/12/27 15:02
  **/
object Tokenizer extends App {

  val spark  = SparkUtils.getSparkSession("Tokenizer",4)
  //sc.setLogLevel("Error")

  //定义数据，每一行堪称一个文档
  val sentenceDF = spark.createDataFrame(Seq(
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
  )).toDF("label", "sentence")

  sentenceDF.show(false)
  /**
    * +-----+-----------------------------------+
    * |label|sentence                           |
    * +-----+-----------------------------------+
    * |0    |Hi I heard about Spark             |
    * |1    |I wish Java could use case classes |
    * |2    |Logistic,regression,models,are,neat|
    * +-----+-----------------------------------+
    */

  //定义默认分词器设置输入输出字段
  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  //使用分词器（转换器）对数据进行转换
  val tokenized = tokenizer.transform(sentenceDF)
  //查看转换后的结果
  tokenized.select("words", "label").take(3).foreach(println)
  /**
    * [WrappedArray(hi, i, heard, about, spark),0]
    * [WrappedArray(i, wish, java, could, use, case, classes),1]
    * [WrappedArray(logistic,regression,models,are,neat),2]
    */
  tokenized.show(false)

  /**
    * +-----+-----------------------------------+------------------------------------------+
    * |label|sentence                           |words                                     |
    * +-----+-----------------------------------+------------------------------------------+
    * |0    |Hi I heard about Spark             |[hi, i, heard, about, spark]              |
    * |1    |I wish Java could use case classes |[i, wish, java, could, use, case, classes]|
    * |2    |Logistic,regression,models,are,neat|[logistic,regression,models,are,neat]     |
    * +-----+-----------------------------------+------------------------------------------+
    */

  //定义一个正则分词器并设置输入输出与分隔符
  val regexTokenizer = new RegexTokenizer()
    .setInputCol("sentence")
    .setOutputCol("words")
    .setPattern("\\W") // 或者使用： .setPattern("\\w+").setGaps(false)


  //使用正则分词器进行转换
  val regexTokenized = regexTokenizer.transform(sentenceDF)
  //查看转换后的数据
  regexTokenized.select("words", "label").take(3).foreach(println)
  /**
    * [WrappedArray(hi, i, heard, about, spark),0]
    * [WrappedArray(i, wish, java, could, use, case, classes),1]
    * [WrappedArray(logistic, regression, models, are, neat),2]
    */
  regexTokenized.show(false)
  /**
    * +-----+-----------------------------------+------------------------------------------+
    * |label|sentence                           |words                                     |
    * +-----+-----------------------------------+------------------------------------------+
    * |0    |Hi I heard about Spark             |[hi, i, heard, about, spark]              |
    * |1    |I wish Java could use case classes |[i, wish, java, could, use, case, classes]|
    * |2    |Logistic,regression,models,are,neat|[logistic, regression, models, are, neat] |
    * +-----+-----------------------------------+------------------------------------------+
    */

}
