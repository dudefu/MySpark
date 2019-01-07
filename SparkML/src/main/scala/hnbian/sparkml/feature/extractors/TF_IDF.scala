package hnbian.spark.ml.feature.extractors

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
  * @author hnbian
  * @ Description
  * @ date 2018/12/26 14:46
  **/
object TF_IDF extends App {
  val conf = new SparkConf().setAppName("TF_IDF")
  //设置master local[4] 指定本地模式开启模拟worker线程数
  conf.setMaster("local[4]")
  //创建sparkContext文件
  val sc = new SparkContext(conf)
  val spark = SparkSession.builder().getOrCreate()
  sc.setLogLevel("Error")
  //开启隐式转换
  import spark.implicits._

  //创建一个DataFrame ，每一个句子代表一个文档。

  val sentenceData = spark.createDataFrame(Seq(
    (0,"I heard about Spark and If love Spark"),
    (0,"I wish Java could Spark use case classes"),
    (1,"Logistic regression Spark models are neat"),
  )).toDF("label","sentence")

  //查看我们刚刚创建的DataFrame
  sentenceData.show(false)
  /**
    * +-----+------------------------------------+
    * |label|sentence                            |
    * +-----+------------------------------------+
    * |0    |I heard about Spark and I love Spark|
    * |0    |I wish Java could use case classes  |
    * |1    |Logistic regression models are neat |
    * +-----+------------------------------------+
    */

  //使用分词器对句子进行分词
  //1.创建分词器 （转换器）
  val tokenizer = new Tokenizer()
    .setInputCol("sentence")
    .setOutputCol("words")
  //2.调用转换器的transform()方法 对句子进行分词
  val wordsData = tokenizer.transform(sentenceData)
  //查看分词器执行结果
  wordsData.show(false)

  /**
    * +-----+------------------------------------+---------------------------------------------+
    * |label|sentence                            |words                                        |
    * +-----+------------------------------------+---------------------------------------------+
    * |0    |I heard about Spark and I love Spark|[i, heard, about, spark, and, i, love, spark]|
    * |0    |I wish Java could use case classes  |[i, wish, java, could, use, case, classes]   |
    * |1    |Logistic regression models are neat |[logistic, regression, models, are, neat]    |
    * +-----+------------------------------------+---------------------------------------------+
    */

  //使用HashingTF 调用transform()方法，把分出来的词转换成特征向量
  //1.定义HashingTF
  val hashingTF = new HashingTF()
    .setInputCol("words")
    .setOutputCol("newFeatures")
    .setNumFeatures(2000) // 设置哈希表桶数为2000 （默认2^20 = 1,048,576）

  //2.调用HashingTF.tronsform()方法 转换特征向量
  val featurizedData = hashingTF.transform(wordsData)

  //查看转换特征向量后的数据
  featurizedData.show(false)

  /**
    * +-----+-----------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------+
    * |label|sentence                                 |words                                            |newFeatures 前面数组是将词转成向量，后面数组是对应下表位置词语在所在文档出现次数  |
    * +-----+-----------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------+
    * |0    |I heard about Spark and If love Spark    |[i, heard, about, spark, and, if, love, spark]   |(2000,[170,240,333,1105,1329,1357,1777],[1.0,1.0,1.0,2.0,1.0,1.0,1.0])        |
    * |0    |I wish Java could Spark use case classes |[i, wish, java, could, spark, use, case, classes]|(2000,[213,342,489,495,1105,1329,1809,1967],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
    * |1    |Logistic regression Spark models are neat|[logistic, regression, spark, models, are, neat] |(2000,[286,695,1105,1138,1193,1604],[1.0,1.0,1.0,1.0,1.0,1.0])                |
    * +-----+-----------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------+
    */

  //使用 IDF对单纯词频特征向量进行修正，使其更能体现不同词汇文本的区别能力，
  //IDF 是一个Estimator，调用fit(）方法并将词频传入，然后产生一个IDFModel
  //1.定义一个IDF
  val idf = new IDF().setInputCol("newFeatures").setOutputCol("features")

  //2.调用IDF.fit()方法，产生一个model
  val idfModel = idf.fit(featurizedData)

  //idfModel是一个transformer调用它的transform(）方法可以得到每个单词对应的TF-IDF度量值
  val rescaledData = idfModel.transform(featurizedData)
  rescaledData.show(false)

  /**
    *
    * +-----+-----------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |label|sentence                                 |words                                            |newFeatures                                                                   |features                                                                                                                                                                                |
    * +-----+-----------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * |0    |I heard about Spark and If love Spark    |[i, heard, about, spark, and, if, love, spark]   |(2000,[170,240,333,1105,1329,1357,1777],[1.0,1.0,1.0,2.0,1.0,1.0,1.0])        |(2000,[170,240,333,1105,1329,1357,1777],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.0,0.28768207245178085,0.6931471805599453,0.6931471805599453])                       |
    * |0    |I wish Java could Spark use case classes |[i, wish, java, could, spark, use, case, classes]|(2000,[213,342,489,495,1105,1329,1809,1967],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|(2000,[213,342,489,495,1105,1329,1809,1967],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.0,0.28768207245178085,0.6931471805599453,0.6931471805599453])|
    * |1    |Logistic regression Spark models are neat|[logistic, regression, spark, models, are, neat] |(2000,[286,695,1105,1138,1193,1604],[1.0,1.0,1.0,1.0,1.0,1.0])                |(2000,[286,695,1105,1138,1193,1604],[0.6931471805599453,0.6931471805599453,0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453])                                               |
    * +-----+-----------------------------------------+-------------------------------------------------+------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    * spark 在所有文档都出现过故值为0
    */

  rescaledData.select("features", "label").take(3).foreach(println)
  /**
    * [(2000,[170,240,333,1105,1329,1357,1777],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.0,0.28768207245178085,0.6931471805599453,0.6931471805599453]),0]
    * [(2000,[213,342,489,495,1105,1329,1809,1967],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.0,0.28768207245178085,0.6931471805599453,0.6931471805599453]),0]
    * [(2000,[286,695,1105,1138,1193,1604],[0.6931471805599453,0.6931471805599453,0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453]),1]
    */
}
