package hnbian.sparkml.classnews

import hnbian.spark.utils.SparkUtils
import hnbian.sparkml.classnews.preprocess.Preprocessor
import hnbian.sparkml.utils.Evaluations
import org.apache.spark.sql.DataFrame


/**
  * @author hnbian
  *         @ Description  模型训练与数据预测与模型评估代码
  *         @ Date 2019/1/7 16:06
  **/
object LRClassPredictDemo {

  /** main 方法 */
  def main(args: Array[String]): Unit = {
    // 训练与保存模型
    train()

    // 预测数据与评估模型
    //predict()
  }

  /**
    * 模型训练并对模型进行评估
    */
  def train(): Unit = {
    val spark = SparkUtils.getSparkSession("LRClassPredictDemoTrain", 4)
    val preprocessor = new Preprocessor
    //特征提取转换 (清洗、分词、向量化)
    val trainDF = preprocessor.train(spark)._1

    //模型训练
    val lrClassifier = new LRClassifier
    val predictions = lrClassifier.train(trainDF)
    //打印模型预测结果相关指标
    evaluation(predictions)

    spark.stop()
  }

  /**
    * 预测数据与查看测试结果
    */
  def predict(): Unit = {
    val spark = SparkUtils.getSparkSession("LRClassPredictDemoPredict", 4)
    //=== 特征提取转换 (清洗、分词、向量化)
    val preprocessor = new Preprocessor
    val (predictDF, indexModel, _) = preprocessor.predict(spark)

    val lrClassifier = new LRClassifier
    val predictions = lrClassifier.predict(predictDF, indexModel)
    //打印模型预测结果
    evaluation(predictions)
    spark.stop()
  }

  /**
    * 模型评估
    * @param predictions
    */
  def evaluation(predictions: DataFrame): Unit = {
    val (accuracy, precision, recall, f1) = Evaluations.multiClassEvaluate(predictions)
    println("\n\n========= 评估结果 ==========")
    println(s"\n准确率：$accuracy")
    println(s"加权精确率：$precision")
    println(s"加权召回率：$recall")
    println(s"F1值：$f1")
    predictions.show(5, false)
  }
}
