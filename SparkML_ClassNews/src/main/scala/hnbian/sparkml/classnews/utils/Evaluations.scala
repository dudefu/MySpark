package hnbian.sparkml.classnews.utils

import hnbian.sparkml.classnews.LRClassPredictDemo.predictions
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Row}


/**
  * @author hnbian
  *         @ Description 多分类结果评估
  *         @ Date 2019/1/7 16:14
  **/
object Evaluations {
  /**
    * 评估结果
    *
    * @param data 分类结果
    * @return (准确率, 召回率, F1)
    */
  def multiClassEvaluate(data: DataFrame): (Double, Double, Double, Double) = {

    val metrics = new MulticlassMetrics(
      predictions.select("prediction", "indexedLabel")
        .rdd.map {
        case Row(prediction: Double, label: Double) => (prediction, label)
      }
    )

    /** 准确率 */
    val accuracy = metrics.accuracy
    /** 精确率 */
    val weightedPrecision = metrics.weightedPrecision
    /** 召回率 */
    val weightedRecall = metrics.weightedRecall
    /** F1 */
    val f1 = metrics.weightedFMeasure

    (accuracy, weightedPrecision, weightedRecall, f1)
  }
}
