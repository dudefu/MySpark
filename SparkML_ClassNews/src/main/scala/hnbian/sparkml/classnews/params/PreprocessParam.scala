package hnbian.sparkml.classnews.params

import hnbian.sparkml.classnews.utils.Conf

import scala.collection.mutable


/**
  * @author hnbian
  * @ Description 对数据预处理过程中使用参数
  * @ Date 2019/1/7 14:50
  **/
class PreprocessParam extends Serializable {

  //val kvMap: mutable.LinkedHashMap[String, String] = Conf.loadConf("src/main/resources/preprocess.properties")
  val kvMap: mutable.LinkedHashMap[String, String] = Conf.loadConf("preprocess.properties")

  val stopwordFilePath: String = kvMap.getOrElse("stopword.file.path", "dictionaries/hanlp/data/dictionary/stopwords.txt")    //停用词表路径
  val segmentType: String = kvMap.getOrElse("segment.type", "StandardSegment")    //分词方式
  val delNum: Boolean = kvMap.getOrElse("is.delete.number", "false").toBoolean     //是否去除数字
  val delEn: Boolean = kvMap.getOrElse("is.delete.english", "false").toBoolean    //是否去除英语单词
  val addNature: Boolean = kvMap.getOrElse("add.nature", "false").toBoolean   //是否添加词性
  val minTermLen: Int = kvMap.getOrElse("min.term.len", "1").toInt      //最小词长度
  val minTermNum: Int = kvMap.getOrElse("min.term.num", "3").toInt      //行最小词数
  val vocabSize: Int = kvMap.getOrElse("vocab.size", "10000").toInt   //特征词汇表大小

  val indexModelPath: String = kvMap.getOrElse("model.index.path", "models/preprocession/indexModel")    //索引模型保存路径
  val vecModelPath: String = kvMap.getOrElse("model.vectorize.path", "models/preprocession/vecModel")   //向量模型保存路径

  /**训练数据保存路径*/
  val dataTrainPath: String = kvMap.getOrElse("data.train.path", "/SparkML_ClassNews/src/main/resources/train/*")

  /**预测数据保存路径*/
  val dataPredictPath: String = kvMap.getOrElse("data.predict.path", "/SparkML_ClassNews/src/main/resources/predict/*")
}
