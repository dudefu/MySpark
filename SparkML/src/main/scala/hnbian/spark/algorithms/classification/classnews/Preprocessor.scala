package hnbian.spark.algorithms.classification.classnews

import com.hankcs.hanlp.tokenizer.lexical.Segmenter
import org.apache.spark.ml.feature.{CountVectorizerModel, StringIndexer, StringIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import params.PreprocessParam

/**
  * @author hnbian
  *         @ Description 特征预处理器
  *         @ Date 2019/1/7 12:01
  **/
class Preprocessor extends Serializable {

  /**
    * 训练数据的预处理方法
    *
    * @param filePath 数据路径
    * @param spark    sparkSession
    * @return （预处理后的数据, 索引模型, 向量模型),
    *         数据包括字段: "label", "indexedLabel", "title", "time", "content", "tokens", "removed", "features"
    */
  def train(filePath: String, spark: SparkSession): (DataFrame, StringIndexerModel, CountVectorizerModel) = {

    null
  }

  /**
    * 用于预测数据的预处理
    *
    * @param filePath 数据路径
    * @param spark    SparkSession
    * @return （预处理后的数据, 索引模型, 向量模型),
    *         数据包括字段: "label", "indexedLabel", "title", "time", "content", "tokens", "removed", "features"
    */
  def predict(filePath: String, spark: SparkSession): (DataFrame, StringIndexerModel, CountVectorizerModel) = {

    null
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

      if (fields.length > 3) {
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
        //println(s"title=${title},title=${time},content=${content}")
        if (!label.equals("其他")) {
          Some(label, title, time, content)
        } else {
          None
        }
      } else None
    }.toDF("label", "title", "time", "content")

    textDF.show()
    textDF
  }

  /**
    * 将字符串label转换为索引形式
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
      .isDelEn(params.delEn)
      .isDelNum(params.delNum)
      .setSegmentType(params.segmentType)
      .addNature(params.addNature)
      .setMinTermLen(params.minTermLen)
      .setMinTermNum(params.minTermNum)
      .setInputCol("content")
      .setOutputCol("tokens")

    val segDF = segmenter.transform(data)


    //=== 去除停用词
    val stopwordArray = spark.sparkContext.textFile(params.stopwordFilePath).collect()
    val remover = new StopWordsRemover()
      .setStopWords(stopwordArray)
      .setInputCol(segmenter.getOutputCol)
      .setOutputCol("removed")
    val removedDF = remover.transform(segDF)

    removedDF
  }

}
