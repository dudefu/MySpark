package hnbian.spark.ml.pipeline

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * @author hnbian
  *         Description pipeline 逻辑回归 代码示例
  *         Date 2018/12/26 10:13
  **/
object PipelineLogisticRegressionDemo extends App {

  val conf = new SparkConf().setAppName("PipelineLogisticRegressionDemo")
  //设置master local[4] 指定本地模式开启模拟worker线程数
  conf.setMaster("local[4]")
  //创建sparkContext文件
  val sc = new SparkContext(conf)
  val spark = SparkSession.builder().getOrCreate()
  sc.setLogLevel("Error")


  //构建训练数据集
  val training = spark.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0)
  )).toDF("id", "text", "label")

  //查看训练数据
  training.show()
  /**
    * +---+----------------+-----+
    * | id|            text|label|
    * +---+----------------+-----+
    * |  0| a b c d e spark|  1.0|
    * |  1|             b d|  0.0|
    * |  2|     spark f g h|  1.0|
    * |  3|hadoop mapreduce|  0.0|
    * +---+----------------+-----+
    */

  //在下面我们要定义 Pipeline 中的各个工作流阶段PipelineStage，
  // 包括转换器和评估器，具体的，
  // 包含tokenizer, hashingTF和lr三个步骤。

  //定义分词器 （转换器）
  val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")

  //定义Hash 处理 (转换器)
  val hashingTF = new HashingTF()
    .setNumFeatures(1000)
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")

  //定义 逻辑回归（评估器）
  val lr = new LogisticRegression()
    .setMaxIter(10).setRegParam(0.01)

  //按照具体的处理逻辑有序的组织PipelineStages 并创建一个Pipeline。
  //构建的Pipeline本质上是一个Estimator，在它的fit（）方法运行之后，它将产生一个PipelineModel，它是一个Transformer。
  val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))

  //训练数据 得到模型
  val model = pipeline.fit(training)

  //可以将工作流保存到磁盘
  //model.write.overwrite().save("/tmp/spark-logistic-regression-model")

  //构建测试数据
  val test = spark.createDataFrame(Seq(
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "spark a"),
    (7L, "apache hadoop")
  )).toDF("id", "text")

  // 并在生产过程中重新加载
  //val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")

  //调用我们训练好的PipelineModel的transform（）方法，让测试数据按顺序通过拟合的工作流，生成我们所需要的预测结果。
  model.transform(test).show(false)
  /**
    * +---+-------------+----------------+------------------------------------------+------------------------------------------+----------------------------------------+----------+
    * |id |text         |words           |features                                  |rawPrediction                             |probability两个元素分别是预测为0和1的概率 |prediction|
    * +---+-------------+----------------+------------------------------------------+------------------------------------------+----------------------------------------+----------+
    * |4  |spark i j k  |[spark, i, j, k]|(1000,[105,149,329,456],[1.0,1.0,1.0,1.0])|[0.16293291377589236,-0.16293291377589236]|[0.5406433544852326,0.4593566455147674] |0.0       |
    * |5  |l m n        |[l, m, n]       |(1000,[6,638,655],[1.0,1.0,1.0])          |[2.6407449286804225,-2.6407449286804225]  |[0.933438262738353,0.06656173726164705] |0.0       |
    * |6  |spark a      |[spark, a]      |(1000,[105,170],[1.0,1.0])                |[-1.7313553283508463,1.7313553283508463]  |[0.15041430048073343,0.8495856995192667]|1.0       |
    * |7  |apache hadoop|[apache, hadoop]|(1000,[181,495],[1.0,1.0])                |[3.7429405136496934,-3.7429405136496934]  |[0.9768636139518375,0.02313638604816234]|0.0       |
    * +---+-------------+----------------+------------------------------------------+------------------------------------------+----------------------------------------+----------+
    */

  model.transform(test).
    select("id", "text", "probability", "prediction").
    collect().
    foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
      println(s"($id, $text) --> prob=$prob, prediction=$prediction")
    }
  /**
    * (4, spark i j k) --> prob=[0.5406433544852323,0.4593566455147678], prediction=0.0
    * (5, l m n) --> prob=[0.933438262738353,0.06656173726164712], prediction=0.0
    * (6, spark a) --> prob=[0.15041430048073337,0.8495856995192667], prediction=1.0
    * (7, apache hadoop) --> prob=[0.9768636139518375,0.02313638604816236], prediction=0.0
    */

  /**
    * 从上面结果我们可以看到，4与6 中都包含“Spark” 其中6 预测为1 与我们的预期相一致，
    * 但是4 预测依然是0，但是通过概率我们看到有46% 的概率预测是1，而5、7 预测为1的概率分别是7%和2%。
    * 这是由于训练数据集较少，如果有更多测试数据那么预测准确率将会有显著提升
    */
}
