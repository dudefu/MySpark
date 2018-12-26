package hnbian.spark.ml.pipeline

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap


/**
  * 评估器、转换器、参数 代码示例
  */
object EstimatorTransformerParamExam {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("EstimatorTransformerParamExam")
    //设置master local[4] 指定本地模式开启模拟worker线程数
    conf.setMaster("local[4]")
    //创建sparkContext文件
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().getOrCreate()
    sc.setLogLevel("Error")
    // 从（标签，特征）元组列表中准备训练数据。
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

    training.show()

    // 创建一个LogisticRegression实例。 这个实例是一个Estimator。
    val lr = new LogisticRegression()
    // 打印出参数，文档和任何默认值
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    // 我们可以使用setter方法来设置参数。
    lr.setMaxIter(5) //设置最大迭代次数。默认100
      .setRegParam(0.01) //设置正则化参数。默认0,对于防止过拟合的调整参数。

    // 学习LogisticRegression模型。 这使用存储在lr中的参数。
    val model1 = lr.fit(training)

    /**
      * model1 是一个模型（由评估器生成的转换器）
      * 下面打印出在fig()方法被调用时使用的参数值
      * LogisticRegression实例
      */
    println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)
    //model1.parent方法返回产生这个模型的Estimator实例（这里就是lr对象），此处打印所有lr的参数当前值

    /**
      * 打印参数值如下：
      * Model 1 was fit using parameters: {
      * logreg_6e008f8d0096-aggregationDepth: 2,
      * logreg_6e008f8d0096-elasticNetParam: 0.0,
      * logreg_6e008f8d0096-family: auto,
      * logreg_6e008f8d0096-featuresCol: features,
      * logreg_6e008f8d0096-fitIntercept: true,
      * logreg_6e008f8d0096-labelCol: label,
      * logreg_6e008f8d0096-maxIter: 5, -- 我们在前面设置的最大迭代次数
      * logreg_6e008f8d0096-predictionCol: prediction,
      * logreg_6e008f8d0096-probabilityCol: probability,
      * logreg_6e008f8d0096-rawPredictionCol: rawPrediction,
      * logreg_6e008f8d0096-regParam: 0.01, -- 我们在前面设置的正则化参数
      * logreg_6e008f8d0096-standardization: true,
      * logreg_6e008f8d0096-threshold: 0.5,
      * logreg_6e008f8d0096-tol: 1.0E-6
      * }
      */
    /**
      * 我们也可以使用ParamMap指定参数，
      * 它支持几种指定参数的方法。
      */
    val paramMap = ParamMap(lr.maxIter -> 20)
      .put(lr.maxIter, 30) // 指定1个参数。 这会覆盖原来的maxIter。
      .put(lr.regParam -> 0.1, lr.threshold -> 0.55) // 指定多个参数。
    println(paramMap)

    /**
      * 打印我们刚刚设置的参数map
      * {
      * logreg_4c341510b992-maxIter: 30,
      * logreg_4c341510b992-regParam: 0.1,
      * logreg_4c341510b992-threshold: 0.55
      * }
      */

    /**
      * 也可以结合ParamMaps
      * 下面增加一个 用于预测类别条件概率的列名 的参数。
      * 注意：并非所有模型都输出经过良好校准的概率估计！ 应将这些概率视为置信度，而不是精确概率。
      */
    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability") // 更改输出列名称，
    val paramMapCombined = paramMap ++ paramMap2 //合并两个参数组
    println(paramMapCombined)

    /**
      * 打印参数
      * {
      * logreg_5bb63eb50d12-maxIter: 30,
      * logreg_5bb63eb50d12-probabilityCol: myProbability,
      * logreg_5bb63eb50d12-regParam: 0.1,
      * logreg_5bb63eb50d12-threshold: 0.55
      * }
      */

    /**
      * 现在使用 paramMapCombined 参数学习一个新模型。
      * paramMapCombined覆盖之前通过lr.set 方法设置的所有参数。
      */
    val model2 = lr.fit(training, paramMapCombined)

    //打印model2使用的参数
    println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

    /**
      * Model 2 was fit using parameters: {
      * logreg_ce73cd6a462f-aggregationDepth: 2,
      * logreg_ce73cd6a462f-elasticNetParam: 0.0,
      * logreg_ce73cd6a462f-family: auto,
      * logreg_ce73cd6a462f-featuresCol: features,
      * logreg_ce73cd6a462f-fitIntercept: true,
      * logreg_ce73cd6a462f-labelCol: label,
      * logreg_ce73cd6a462f-maxIter: 30,
      * logreg_ce73cd6a462f-predictionCol: prediction,
      * logreg_ce73cd6a462f-probabilityCol: myProbability,
      * logreg_ce73cd6a462f-rawPredictionCol: rawPrediction,
      * logreg_ce73cd6a462f-regParam: 0.1,
      * logreg_ce73cd6a462f-standardization: true,
      * logreg_ce73cd6a462f-threshold: 0.55,
      * logreg_ce73cd6a462f-tol: 1.0E-6
      * }
      */

    // 准备测试数据。
    //用model2对test进行转换（只会使用test的特征列，默认为"features"列），会生成预测列、概率列等新列
    val test = spark.createDataFrame(Seq(
      (0.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")
    test.show()

    /**
      * 使用Transformer.transform() 方法对测试数据进行预测。
      * LogisticRegression.transform只会使用'features'列。
      * 注意model2.transform()输出一个'myProbability'列，而不是通常的
      * 'probability'列，因为我们以前重命名了lr.probabilityCol参数。
      *
      */
/*     model2.transform(test)
       .select("features", "label", "myProbability", "prediction")
       .collect()
       .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
         println(s"($features, $label) -> prob=$prob, prediction=$prediction")
       }*/

    model2.transform(test).show(false)
    /**
    |-----+--------------+----------------------------------------+----------------------------------------+----------+
    |label|features      |rawPrediction                           |myProbability                           |prediction|
    |标签 |特征向量       |原始预测                                |预测概率，两个元素分别是预测为0和1的概率   |预测标签  |
    |-----+--------------+----------------------------------------+----------------------------------------+----------+
    |1.0  |[-1.0,1.5,1.3]|[-2.804656941874642,2.804656941874642]  |[0.05707304171034024,0.9429269582896597]|1.0       |
    |0.0  |[3.0,2.0,-0.1]|[2.4958763566420603,-2.4958763566420603]|[0.9238522311704105,0.07614776882958946]|0.0       |
    |1.0  |[0.0,2.2,-1.5]|[-2.093524902791379,2.093524902791379]  |[0.10972776114779474,0.8902722388522052]|1.0       |
    |-----+--------------+----------------------------------------+----------------------------------------+----------+
      */
  }
}
