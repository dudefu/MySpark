package hnbian.sparkml.classnews.preprocess

import java.io.File

import hnbian.sparkml.classnews.params.PreprocessParam
import hnbian.sparkml.classnews.utils.IOUtils
import org.apache.spark.ml.feature._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}


/**
  * @author hnbian
  *         @ Description 特征预处理器
  *         @ Date 2019/1/7 12:01
  **/
class Preprocessor extends Serializable {

  /**
    * 训练数据的预处理方法
    *
    * @param spark    sparkSession
    * @return （预处理后的数据, 索引模型, 向量模型),
    *         数据包括字段: "label", "indexedLabel", "title", "time", "content", "tokens", "removed", "features"
    */
  def train( spark: SparkSession): (DataFrame, StringIndexerModel, CountVectorizerModel) = {
    val params = new PreprocessParam
    val filePath = s"${System.getProperty("user.dir") }${params.dataTrainPath}"
    val cleanDF = this.clean(filePath, spark) //清洗数据
    val indexModel = this.indexrize(cleanDF)
    val indexDF = indexModel.transform(cleanDF) //标签索引化
    //分词、移除停用词
    val segDF = this.segment(indexDF, params) //分词
    //向量化过程, 包括词汇表过滤
    val vecModel = this.vectorize(segDF, params)
    val trainDF = vecModel.transform(segDF) //向量化
    /**
      * 保存预处理模型
      */
    this.saveModel(indexModel, vecModel, params)
    (trainDF, indexModel, vecModel)
  }

  /**
    * 用于预测数据的预处理
    *
    * @param filePath 数据路径
    * @param spark    SparkSession
    * @return （预处理后的数据, 索引模型, 向量模型),
    *         数据包括字段: "label", "indexedLabel", "title", "time", "content", "tokens", "removed", "features"
    */
  def predict( spark: SparkSession): (DataFrame, StringIndexerModel, CountVectorizerModel) = {

    val params = new PreprocessParam
    val filePath = s"${System.getProperty("user.dir") }${params.dataPredictPath}"

    //清洗数据
    val cleanDF = this.clean(filePath, spark)

    //标签索引模型
    val (indexModel, vecModel) = this.loadModel(params)
    val indexDF = indexModel.transform(cleanDF)

    //分词过程，包括"分词", "去除停用词"
    val segDF = this.segment(indexDF, params)
    val predictDF = vecModel.transform(segDF)
    (predictDF, indexModel, vecModel)
  }

  /**
    * 清洗步骤, 可根据具体数据结构和业务场景的不同进行重写. 注意: 输出必须要有标签字段"label"
    *
    * @param filePath 数据路径
    * @param spark    sparkSession
    * @return 清洗后的数据, 包含字段: "label", "title", "time", "content"
    */
  def clean(filePath: String, spark: SparkSession): DataFrame = {
    import spark.implicits._

    val textRDD = spark.sparkContext.textFile(filePath)

    val textDF = textRDD.flatMap { line =>
      //分隔数据
      val fields = line.split("\u00EF")

      if (null != fields && fields.length > 3) {
        val categoryLine = fields(0)
        val categories = categoryLine.split("\\|")
        val category = categories.last

        var label = "其他"
        if (category.contains("文化")) label = "文化"
        else if (category.contains("财经")) label = "财经"
        else if (category.contains("军事")) label = "军事"
        else if (category.contains("体育")) label = "体育"
        else {}

        val (title, time, content) = (fields(1), fields(2), fields(3))
        if (!label.equals("其他")) {
          Some(label, title, time, content)
        } else {
          None
        }
      } else None
    }.toDF("label", "title", "time", "content")

    textDF
  }

  /**
    * 将数据中的中文标签转换为索引形式输出
    *
    * @param data 输入数据
    * @return 标签索引模型, 模型增加字段: "indexedLabel"
    */
  def indexrize(data: DataFrame): StringIndexerModel = {
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    labelIndexer
  }


  /**
    * 分词过程，包括"分词", "去除停用词"
    *
    * @param data   输入数据
    * @param params 配置参数
    * @return 预处理后的DataFrame, 增加字段: "tokens", "removed"
    */
  def segment(data: DataFrame, params: PreprocessParam): DataFrame = {
    val spark = data.sparkSession

    //=== 分词
    val segmenter = new Segmenter()
      .isDelEn(params.delEn) //是否去除英语单词
      .isDelNum(params.delNum) //是否去除数字
      .setSegmentType(params.segmentType) //分词方式
      .addNature(params.addNature) //是否添加词性
      .setMinTermLen(params.minTermLen) //最小词长度
      .setMinTermNum(params.minTermNum) //行最小词数
      .setInputCol("content")
      .setOutputCol("tokens")

    val segDF = segmenter.transform(data)


    //=== 去除停用词
    val stopwordArray = spark.sparkContext.textFile(params.stopwordFilePath).collect()
    val remover = new StopWordsRemover()
      .setStopWords(stopwordArray) //这是停用词集合
      .setInputCol(segmenter.getOutputCol)
      .setOutputCol("removed")
    val removedDF = remover.transform(segDF)

    removedDF
  }

  /**
    * 向量化过程, 包括词汇表过滤
    *
    * @param data   输入数据
    * @param params 配置参数
    * @return 向量模型
    */
  def vectorize(data: DataFrame, params: PreprocessParam): CountVectorizerModel = {
    //=== 向量化
    val vectorizer = new CountVectorizer()
      .setVocabSize(params.vocabSize) //特征词汇表大小
      .setInputCol("removed")
      .setOutputCol("features")
    val parentVecModel = vectorizer.fit(data)

    //过滤词汇表过
    val numPattern = "[0-9]+".r
    val vocabulary = parentVecModel.vocabulary.flatMap { term =>
      //  1. 滤长度为1的词           2. 过滤数字
      if (term.length == 1 || term.matches(numPattern.regex)) None else Some(term)
    }

    val vecModel = new CountVectorizerModel(Identifiable.randomUID("cntVec"), vocabulary)
      .setInputCol("removed")
      .setOutputCol("features")

    //返回转换器模型
    vecModel.transform(data).select("label","indexedLabel","features").show(false)
    vecModel
  }

  /**
    * 保存预处理模型
    *
    * @param indexModel 标签索引模型
    * @param vecModel   向量模型
    * @param params     配置参数
    */
  def saveModel(indexModel: StringIndexerModel, vecModel: CountVectorizerModel, params: PreprocessParam): Unit = {
    val indexModelPath = params.indexModelPath
    val vecModelPath = params.vecModelPath

    val indexModelFile = new File(indexModelPath)
    val vecModelFile = new File(vecModelPath)


    if (indexModelFile.exists()) {
      println("索引模型已存在，新模型将覆盖原有模型...")
      IOUtils.delDir(indexModelFile)
    }
    if (vecModelFile.exists()) {
      println("向量模型已存在，新模型将覆盖原有模型...")
      IOUtils.delDir(vecModelFile)
    }

    indexModel.save(indexModelPath)
    vecModel.save(vecModelPath)
    println("预处理模型已保存！")
  }


  /**
    * 加载预处理模型
    *
    * @param params 配置参数
    * @return LR模型
    */
  def loadModel(params: PreprocessParam): (StringIndexerModel, CountVectorizerModel) = {
    val indexModelPath = params.indexModelPath
    val vecModelPath = params.vecModelPath

    val indexModelFile = new File(indexModelPath)
    val vecModelFile = new File(vecModelPath)

    if (!indexModelFile.exists()) {
      println("索引模型不存在，即将退出！")
      System.exit(1)
    } else if (!vecModelFile.exists()) {
      println("向量模型不存在，即将退出！")
      System.exit(1)
    } else {
      println("开始加载预处理模型...")
    }

    val indexModel = StringIndexerModel.load(indexModelPath)
    val vecModel = CountVectorizerModel.load(vecModelPath)
    println("预处理模型加载成功！")

    (indexModel, vecModel)
  }

}
