package hnbian.sparkml.association

import hnbian.spark.utils.SparkUtils
import org.apache.spark.mllib.fpm.AssociationRules.Rule
import org.apache.spark.mllib.fpm.FPGrowth
import utils.FileUtils
//import org.apache.spark.ml.fpm.FPGrowth

/**
  * @author hnbian 2019/4/17 14:31
  *         关联规则代码示例
  */
object FPGrowth_mllib extends App {
  val spark = SparkUtils.getSparkSession("FPGrowth", 4)

  import spark.implicits._

  //最小支持度
  val minSupport = 0.055
  //最小置信度
  val minConfidence = 0.0001
  //数据分区
  val numPartitions = 10

  //取出数据
  val filePath = FileUtils.getFilePath("association_rules.txt")
  println(filePath)
  val data = spark.sparkContext.textFile(filePath)
  //把数据通过空格分割
  val transactions = data.map(x => x.split(","))
  val data_count = data.collect().length
  println("length==>" + data_count)
  transactions.persist()
  val transactionsDF = transactions.toDF("items")
  //transactionsDF.show()


  //使用  sparkmllib包下面的算法
  //创建一个FPGrowth 算法实例
  val fpg = new FPGrowth()

  //设置训练时候的最小支持度和分区数据
  fpg.setMinSupport(minSupport) //默认0.3
  fpg.setNumPartitions(numPartitions)

  //把数据带入算法中
  val model = fpg.run(transactions)
  println("打印频繁项集出现次数")
  //查看所有频繁项集， 并列出他们出现的次数
  val support_rdd = model.freqItemsets.filter(itemset => {
    itemset.items.length == 2 //选择过滤出几项数据
  }).map(itemset => {
    (itemset.items.mkString(","), itemset.freq, itemset.freq.toDouble / data_count) //itemset.freq 出现次数
  })


  val bb = model.freqItemsets.persist()
  bb.filter(x=>x.items.length ==3 ).foreach(f=>{
    println(f.items.toBuffer)
  })
  val candidates = bb.flatMap { itemset =>
    val items = itemset.items
    println(items)
    items.flatMap { item =>
      items.partition(_ == item) match {
        case (consequent, antecedent) if !antecedent.isEmpty =>
          Some((antecedent.toSeq, (consequent.toSeq, itemset.freq)))
        case _ => None
      }
    }
  }
  candidates.toDF("k","v").show(1000000)


  val support_df = support_rdd.toDF("items", "count", "support")
  //support_df.orderBy(support_df("items").asc).show(100) //按照频繁项集出现的次数排序


  println("打印置信度")
  // 通过置信度筛选出推荐规则，
  //antecedent 表示前项
  //consequent 表示后项
  val confidence_rdd = model.generateAssociationRules(minConfidence)
  /*  .filter(
      x => {(x.antecedent.length==3 && x.consequent.length == 1)} //过滤出前项长度为3 并且后项长度为1 的条件
    )*/
    .map(x => {
    //println("前项"+x.antecedent.toSet.toBuffer+",后项"+x.consequent.toSet.toBuffer+"=, 置信度",x.confidence)
    (x.antecedent.mkString(","), x.consequent.mkString(","),x.confidence)
  })

  val confidence_df = confidence_rdd.toDF("antecedent", "consequent", "confidence")
  //confidence_df.show(100)



  ////查看所有频繁项集， 并列出他们出现的次数
 /* val support_rdd2 = model.freqItemsets.map(itemset => {
    (itemset.items.mkString(","), itemset.freq, itemset.freq.toDouble / data_count)
  })
  support_rdd2.toDF().show()*/

/*  val support_df2 = support_rdd2.toDF("items", "count", "support")
  support_df2.show(100, false)*/

  /*println("打印置信度")
  // 通过置信度筛选出推荐规则，
  //antecedent 表示前项
  //consequent 表示后项
  val confidence_rdd = model.generateAssociationRules(minConfidence).map(x => {
    //println("前项"+x.antecedent.toSet.toBuffer+",后项"+x.consequent.toSet.toBuffer+"=, 置信度",x.confidence)
    (x.antecedent.mkString(","), x.consequent.mkString(","),x.confidence)
  })

  val confidence_df = confidence_rdd.toDF("antecedent", "consequent", "confidence")
  confidence_df.show()
  //,confidence_df("confidence")/support_df("support")
   val df = confidence_df.join(support_df, confidence_df("consequent") === support_df("items"), "left")
     .selectExpr("antecedent", "consequent", "support as consequent_support", "confidence", "confidence/support as lift")
     .filter("lift > 2")
     .orderBy("lift")
   df.show(100)
   //println("生成规则数量")
   //查看规则生成的数量
   //println(model.generateAssociationRules(minConfidence).collect().length)

 */
}
