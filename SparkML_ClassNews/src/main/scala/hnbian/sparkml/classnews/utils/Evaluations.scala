package hnbian.sparkml.classnews.utils

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD


/**
  * @author hnbian
  * @ Description 多分类结果评估
  * @ Date 2019/1/7 16:14
  **/
object Evaluations {
  /**
    * 评估结果
    *
    * @param data 分类结果
    * @return (准确率, 召回率, F1)
    */
  def multiClassEvaluate(data: RDD[(Double, Double)]): (Double, Double, Double) = {
    val metrics = new MulticlassMetrics(data)
    val weightedPrecision = metrics.weightedPrecision
    val weightedRecall = metrics.weightedRecall
    val f1 = metrics.weightedFMeasure

    (weightedPrecision, weightedRecall, f1)
  }
}
