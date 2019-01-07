package hnbian.spark.ml.feature.extractors

import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.feature.Word2Vec

/**
  * 特征提取 Word2Vec 测试代码
  * @author hnbian
  * @ Description
  * @ Date 2018/12/27 10:47
  **/
object Word2Vec extends App {
  val spark  = SparkUtils.getSparkSession("Word2Vec",4)

  //定义测试数据
  val documentDF = spark.createDataFrame(Seq(
    "Hi I heard about Spark".split(" "),
    "I wish Java could use case classes".split(" "),
    "Logistic regression models are neat".split(" ")
  ).map(Tuple1.apply)).toDF("text")

  documentDF.show(false)
  /**
    * +------------------------------------------+
    * |text                                      |
    * +------------------------------------------+
    * |[Hi, I, heard, about, Spark]              |
    * |[I, wish, Java, could, use, case, classes]|
    * |[Logistic, regression, models, are, neat] |
    * +------------------------------------------+
    */
  /**
    * 定义一个Word2Vec
    * 设置相应的超参数，这里设置特征向量的维度为3，
    * Word2Vec模型还有其他可设置的超参数，具体的超参数描述可以参见
    * http://spark.apache.org/docs/1.6.2/api/scala/index.html#org.apache.spark.ml.feature.Word2Vec
    */
  val word2Vec = new Word2Vec().
           setInputCol("text").
           setOutputCol("result").
           setVectorSize(5).
           setMinCount(0)

  //读入训练数据，调用fit()方法生成一个Word2VecModel
  val model = word2Vec.fit(documentDF)
  //使用Word2VecModel把文档转变成特征向量
  model.transform(documentDF).show(false)
  /**
    * +------------------------------------------+----------------------------------------------------------------+
    * |text                                      |result                                                          |
    * +------------------------------------------+----------------------------------------------------------------+
    * |[Hi, I, heard, about, Spark]              |[-0.008142343163490296,0.02051363289356232,0.03255096450448036] |
    * |[I, wish, Java, could, use, case, classes]|[0.043090314205203734,0.035048123182994974,0.023512658663094044]|
    * |[Logistic, regression, models, are, neat] |[0.038572299480438235,-0.03250147425569594,-0.01552378609776497]|
    * +------------------------------------------+----------------------------------------------------------------+
    */
  //文档被转变为了一个3维的特征向量，这些特征向量就可以被应用到相关的机器学习方法中。
}
