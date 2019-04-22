package hnbian.sparkml.association

import hnbian.spark.utils.SparkUtils
import org.apache.spark.mllib.fpm.FPGrowth
import utils.FileUtils

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
  val minConfidence = 0.2
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
    itemset.items.length >= 3 //选择过滤出几项数据
  }).map(itemset => {
    (itemset.items.mkString(","), itemset.freq, itemset.freq.toDouble / data_count) //itemset.freq 出现次数
  })

  val support_df = support_rdd.toDF("items", "count", "support")
  support_df.orderBy(support_df("items").asc).show(5) //按照频繁项集出现的次数排序
  /**
    * +-----------+-----+-------------------+
    * |      items|count|            support|
    * +-----------+-----+-------------------+
    * |A 1,B 3,C 4|   52|0.05591397849462366|
    * |A 1,C 2,D 4|   52|0.05591397849462366|
    * |A 2,B 4,C 4|   80|0.08602150537634409|
    * |A 2,B 4,H 4|   73|0.07849462365591398|
    * |A 2,C 4,H 4|   76|0.08172043010752689|
    * +-----------+-----+-------------------+
    */
  val bb = model.freqItemsets.persist()
/*  bb.filter(x=>x.items.length ==3 ).foreach(f=>{
    println(f.items.toBuffer) //打印项集元素个数
})*/
  /*val candidates = bb.flatMap { //根据支持度频繁项集 生成置信度前项后项的关系
    itemset =>{
      val items = itemset.items
      if(items.length == 3){
        println("items==>"+items.toBuffer)
      }

      items.flatMap { item =>
        items.partition(_ == item) match {
          case (consequent, antecedent) if !antecedent.isEmpty =>  //if(items.length == 4)println(s"items=${items.toBuffer},item=${item},antecedent=${antecedent.toSeq},consequent=${consequent.toSeq}")
            Some((antecedent.toSeq, (consequent.toSeq, itemset.freq,item)))
          case _ => None
        }
      }
    }
  }
  candidates.toDF("k","v").where("size(k) == 3").show(90)

*/





  println("打印置信度")
  // 通过置信度筛选出推荐规则，
  //antecedent 表示前项
  //consequent 表示后项
  val confidence_rdd = model.generateAssociationRules(minConfidence)
    .filter(
      x => {(x.antecedent.length>=2 && x.consequent.length == 1)} //过滤出前项长度为3 并且后项长度为1 的条件
    )
    .map(x => {
    println(s"前项:${x.antecedent.toSet.toBuffer},后项:${x.consequent.toSet.toBuffer},置信度:${x.confidence},提升度:${x.lift}")
    /**
      * 前项:ArrayBuffer(A 4, B 4),后项:ArrayBuffer(H 4),置信度:0.488,提升度:Some(1.0935903614457831)
      * 前项:ArrayBuffer(C 3, F 3),后项:ArrayBuffer(H 4),置信度:0.875,提升度:Some(1.9608433734939759)
      * 前项:ArrayBuffer(F 3, A 2),后项:ArrayBuffer(H 4),置信度:0.7261904761904762,提升度:Some(1.6273666092943202)
      * ...
      */
    (x.antecedent.mkString(","), x.consequent.mkString(","),x.confidence,x.lift,x.toString())
  })

  val confidence_df = confidence_rdd.toDF("antecedent", "consequent", "confidence","lift","remark")
  confidence_df.orderBy(confidence_df("confidence").desc).show(5,false)
  /**
    * +-----------+----------+------------------+------------------+----------------------------------------------------------------------------------------+
    * |antecedent |consequent|confidence        |lift              |remark                                                                                  |
    * +-----------+----------+------------------+------------------+----------------------------------------------------------------------------------------+
    * |F 3,A 4    |H 4       |0.8795180722891566|1.9709682101901582|{F 3,A 4} => {H 4}: (confidence: 0.8795180722891566; lift: Some(1.9709682101901582))    |
    * |C 3,F 3    |H 4       |0.875             |1.9608433734939759|{C 3,F 3} => {H 4}: (confidence: 0.875; lift: Some(1.9608433734939759))                 |
    * |F 3,B 4    |H 4       |0.8088235294117647|1.8125442948263644|{F 3,B 4} => {H 4}: (confidence: 0.8088235294117647; lift: Some(1.8125442948263644))    |
    * |F 3,C 4    |H 4       |0.7941176470588235|1.7795889440113395|{F 3,C 4} => {H 4}: (confidence: 0.7941176470588235; lift: Some(1.7795889440113395))    |
    * |E 3,B 4,H 4|D 2       |0.7761194029850746|1.9667330920330228|{E 3,B 4,H 4} => {D 2}: (confidence: 0.7761194029850746; lift: Some(1.9667330920330228))|
    * +-----------+----------+------------------+------------------+----------------------------------------------------------------------------------------+
    */
}
