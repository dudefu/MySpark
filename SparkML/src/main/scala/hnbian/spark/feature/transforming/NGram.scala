package hnbian.spark.ml.feature.transforming

import utils.SparkUtils
import org.apache.spark.ml.feature.NGram
/**
  * @author hnbian
  *         @ Description 
  *         @ Date 2018/12/27 16:16
  **/
object NGram extends App {

  val spark = SparkUtils.getSparkSession("NGram",4)
  val wordDF = spark.createDataFrame(Seq(
    (0, Array("Hi", "I", "heard", "about", "Spark")),
    (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
    (2, Array("Logistic", "regression", "models", "are", "neat"))
  )).toDF("label", "words")

  wordDF.show(false)
  /**
    * +-----+------------------------------------------+
    * |label|words                                     |
    * +-----+------------------------------------------+
    * |0    |[Hi, I, heard, about, Spark]              |
    * |1    |[I, wish, Java, could, use, case, classes]|
    * |2    |[Logistic, regression, models, are, neat] |
    * +-----+------------------------------------------+
    */
  val ngram = new NGram()
    .setInputCol("words")
    .setOutputCol("ngrams")
    //.setN(3) //默认2，设置连接单词的个数

  val ngramDF = ngram.transform(wordDF)

  ngramDF.take(3).map(_.getAs[Stream[String]]("ngrams").toList).foreach(println)

  ngramDF.show(false)
  /**
    * +-----+------------------------------------------+------------------------------------------------------------------+
    * |label|words                                     |ngrams                                                            |
    * +-----+------------------------------------------+------------------------------------------------------------------+
    * |0    |[Hi, I, heard, about, Spark]              |[Hi I, I heard, heard about, about Spark]                         |
    * |1    |[I, wish, Java, could, use, case, classes]|[I wish, wish Java, Java could, could use, use case, case classes]|
    * |2    |[Logistic, regression, models, are, neat] |[Logistic regression, regression models, models are, are neat]    |
    * +-----+------------------------------------------+------------------------------------------------------------------+
    */
}
