package hnbian.sparkml.classnews.params

import hnbian.sparkml.classnews.utils.Conf

import scala.collection.mutable


/**
  * @author hnbian
  * @ Description 分类训练/测试使用参数
  * @ Date 2019/1/7 15:59
  **/
class ClassParam  extends Serializable {
  //val kvMap: mutable.LinkedHashMap[String, String] = Conf.loadConf("src/main/resources/classification.properties")
  val kvMap: mutable.LinkedHashMap[String, String] = Conf.loadConf("classification.properties")
  val maxIteration: Int = kvMap.getOrElse("max.iteration", "80").toInt    //模型最大迭代次数
  val regParam: Double = kvMap.getOrElse("reg.param", "0.3").toDouble   //正则化项参数
  val elasticNetParam: Double = kvMap.getOrElse("elastic.net.param", "0.1").toDouble   //L1范式比例, L1/(L1 + L2)
  val converTol: Double = kvMap.getOrElse("conver.tol", "1E-6").toDouble    //模型收敛阈值

  val minInfoGain: Double = kvMap.getOrElse("min.info.gain", "0.0").toDouble    //最小信息增益阈值
  val maxDepth: Int = kvMap.getOrElse("max.depth", "10").toInt    //决策树最大深度

  val modelLRPath: String = kvMap.getOrElse("model.lr.path", "models/classification/lrModel")    //LR模型保存路径
  val modelDTPath: String = kvMap.getOrElse("model.dt.path", "models/classification/dtModel")   //决策树模型保存路径

}
