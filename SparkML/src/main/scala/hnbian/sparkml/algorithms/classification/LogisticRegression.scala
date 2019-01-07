package hnbian.spark.ml.algorithms.classification


import hnbian.spark.utils.SparkUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import utils.FileUtils

/**
  * @author hnbian
  * @ Description 逻辑回归二项式预测
  * @ Date 2019/1/3 15:33
  **/
object LogisticRegression extends App{
  val spark = SparkUtils.getSparkSession("LogisticRegression",4)
  import spark.implicits._

  val filePath = FileUtils.getFilePath("sample_libsvm_data.txt")

  val training = spark.read.format("libsvm").load(filePath)
  val lr = new LogisticRegression()
    .setMaxIter(10) //最大迭代次数
    .setRegParam(0.3) //正则化参数
    .setElasticNetParam(0.8) //正则化范式比(默认0.0)，正则化一般有两种范式：L1(Lasso)和L2(Ridge)。L1一般用于特征的稀疏化，L2一般用于防止过拟合。这里的参数即设置L1范式的占比，默认0.0即只使用L2范式
    //.setFamily("binomial")
    .setThreshold(1.0)

  // Fit the model
  val lrModel = lr.fit(training)

  // 打印逻辑回归的系数和截距
  println(s"Coefficients: ${lrModel.coefficients} \nIntercept: ${lrModel.intercept}")
  //Coefficients: (692,[244,263,272,300,301,328,350,351,378,379,405,406,407,428,433,434,455,456,461,462,483,484,489,490,496,511,512,517,539,540,568],[-7.35398352418814E-5,-9.102738505589432E-5,-1.9467430546904216E-4,-2.030064247348659E-4,-3.1476183314865005E-5,-6.842977602660699E-5,1.5883626898237813E-5,1.4023497091369702E-5,3.5432047524968963E-4,1.1443272898170924E-4,1.0016712383666388E-4,6.014109303795469E-4,2.8402481791227693E-4,-1.1541084736508769E-4,3.8599688631290956E-4,6.350195574241061E-4,-1.1506412384575594E-4,-1.5271865864986703E-4,2.8049338089942207E-4,6.070117471191611E-4,-2.0084596632474318E-4,-1.4210755792901163E-4,2.739010341160889E-4,2.7730456244968185E-4,-9.838027027269304E-5,-3.808522443517673E-4,-2.5315198008554816E-4,2.7747714770754383E-4,-2.443619763919179E-4,-0.0015394744687597863,-2.3073328411331095E-4])
  //Intercept: 0.224563159612503

  /**
    * 获取训练集上的模型摘要。 如果`trainingSummary == None`或它是多类模型，则抛出异常。
    */
  val trainingSummary = lrModel.binarySummary

  // 获得每次迭代的目标
  val objectiveHistory = trainingSummary.objectiveHistory
  objectiveHistory.foreach(loss => println(loss))
  /**
    * objectiveHistory:
    * 0.6833149135741672
    * 0.6662875751473734
    * 0.6217068546034618
    * 0.6127265245887887
    * 0.6060347986802872
    * 0.6031750687571563
    * 0.5969621534836272
    * 0.5940743031983121
    * 0.5906089243339023
    * 0.5894724576491043
    * 0.5882187775729587
    */

  // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
  val roc = trainingSummary.roc
  roc.show()
  /**
    * +---+--------------------+
    * |FPR|                 TPR|
    * +---+--------------------+
    * |0.0|                 0.0|
    * |0.0|0.017543859649122806|
    * |0.0| 0.03508771929824561|
    * |0.0| 0.05263157894736842|
    * |0.0| 0.07017543859649122|
    * |0.0|  0.3157894736842105|
    * |0.0|  0.3333333333333333|
    * +---+--------------------+
    */
  println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

  // 设置模型阈值以最大化F-Measure
  val fMeasure = trainingSummary.fMeasureByThreshold
  fMeasure.show(false)
  /**
    * +------------------+--------------------+
    * |threshold         |F-Measure           |
    * +------------------+--------------------+
    * |0.7845860015371141|0.034482758620689655|
    * |0.784319334416892 |0.06779661016949151 |
    * |0.784297609251013 |0.1                 |
    * |0.7842531051133191|0.13114754098360656 |
    * |0.7788060694625323|0.45945945945945943 |
    * |0.7783754276111222|0.4799999999999999  |
    * |0.7771658291080573|0.5                 |
    * |0.7769914303593917|0.5194805194805194  |
    * +------------------+--------------------+
    */

  import org.apache.spark.sql.functions.max
  val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
  println(s"maxFMeasure: ${maxFMeasure}")
  //maxFMeasure: 1.0

  //获取最大阈值
  val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
    .select("threshold").head().getDouble(0)
  //设置阈值
  lrModel.setThreshold(bestThreshold)
  //通过模型得到预测结果
  val predictions = lrModel.transform(training)
  //打印预测结果
  predictions.select("label","prediction","rawPrediction","probability").show(20,false)

  /**
    * +-----+----------+------------------------------------------+----------------------------------------+
    * |label|prediction|rawPrediction                             |probability                             |
    * +-----+----------+------------------------------------------+----------------------------------------+
    * |1.0  |1.0       |[-1.2006427604100238,1.2006427604100238]  |[0.23136089273325805,0.7686391072667419]|
    * |1.0  |1.0       |[-0.9725809559312681,0.9725809559312681]  |[0.27436636195954833,0.7256336380404517]|
    * |1.0  |1.0       |[-1.0780500487239726,1.0780500487239726]  |[0.2538752041717863,0.7461247958282137] |
    * |1.0  |1.0       |[-1.08453337526972,1.08453337526972]      |[0.2526490765347195,0.7473509234652804] |
    * |0.0  |0.0       |[0.7376543954891006,-0.7376543954891006]  |[0.6764827243160596,0.32351727568394034]|
    * |1.0  |1.0       |[-1.2286964884747311,1.2286964884747311]  |[0.2264096521620533,0.7735903478379467] |
    * |1.0  |1.0       |[-1.259664579572604,1.259664579572604]    |[0.22103163838285006,0.77896836161715]  |
    * |1.0  |1.0       |[-1.2371063245185787,1.2371063245185787]  |[0.22494007343582254,0.7750599265641774]|
    * |0.0  |0.0       |[0.738396178597871,-0.738396178597871]    |[0.6766450451466368,0.32335495485336313]|
    * |1.0  |1.0       |[-1.2123284339889662,1.2123284339889662]  |[0.2292893207049596,0.7707106792950403] |
    * |1.0  |0.0       |[-0.23508568050538953,0.23508568050538953]|[0.4414977605721645,0.5585022394278355] |
    * +-----+----------+------------------------------------------+----------------------------------------+
    */

  //模型评估

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  //计算错误率
  val accuracy = evaluator.evaluate(predictions)
  //打印准确率
  println(s"准确率：${accuracy}")
  //打印错误率
  println("Test Error = " + (1.0 - accuracy))
  //Test Error = 0.010000000000000009

  spark.stop()
}
