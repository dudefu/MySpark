package hnbian.sparkml.classnews
import java.io.File


import hnbian.sparkml.classnews.params.ClassParam
import hnbian.sparkml.classnews.utils.IOUtils
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
import org.apache.spark.sql.DataFrame


/**
  * @author hnbian
  * @ Description 逻辑回归多分类
  * @ Date 2019/1/7 16:03
  **/
class LRClassifier {
  /**
  * LR模型训练处理过程, 包括: "模型训练"
  *
  * @param data   训练集
  * @return (向量模型, LR模型)
  */
  def train(data: DataFrame): LogisticRegressionModel = {
    val params = new ClassParam

    //=== LR分类模型训练
    data.persist()
    val lrModel = new LogisticRegression()
      .setMaxIter(params.maxIteration)  //模型最大迭代次数
      .setRegParam(params.regParam) //正则化项参数
      .setElasticNetParam(params.elasticNetParam) //L1范式比例, L1/(L1 + L2)
      .setTol(params.converTol) //模型收敛阈值
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .fit(data)
    data.unpersist()
    this.saveModel(lrModel, params)
    lrModel.transform(data).select("label","prediction","probability").show(false)
    lrModel
  }


  /**
    * LR预测过程, 包括"LR预测", "模型评估"
    *
    * @param data     测试集
    * @return 预测DataFrame, 增加字段:"rawPrediction", "probability", "prediction", "predictedLabel"
    */
  def predict(data: DataFrame, indexModel: StringIndexerModel): DataFrame = {
    val params = new ClassParam
    val lrModel = this.loadModel(params)

    //=== LR预测
    val predictions = lrModel.transform(data)

    //=== 索引转换为label
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(indexModel.labels)
    val result = labelConverter.transform(predictions)
    result
  }


  /**
    * 保存模型
    *
    * @param lrModel  LR模型
    * @param params 配置参数
    */
  def saveModel(lrModel: LogisticRegressionModel, params: ClassParam): Unit = {
    val filePath = params.modelLRPath
    val file = new File(filePath)
    if (file.exists()) {
      println("LR模型已存在，新模型将覆盖原有模型...")
      IOUtils.delDir(file)
    }

    lrModel.save(filePath)
    println("LR模型已保存！")
  }


  /**
    * 加载模型
    *
    * @param params 配置参数
    * @return LR模型
    */
  def loadModel(params: ClassParam): LogisticRegressionModel = {
    val filePath = params.modelLRPath
    val file = new File(filePath)
    if (!file.exists()) {
      println("LR模型不存在，即将退出！")
      System.exit(1)
    } else {
      println("开始加载LR模型...")
    }

    val lrModel = LogisticRegressionModel.load(filePath)
    println("LR模型加载成功！")

    lrModel
  }
}
