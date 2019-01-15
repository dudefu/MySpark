package hnbian.sparksql.udaf

import org.apache.spark.sql.{Row}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import hnbian.spark.utils.SparkUtils


/**
  * @author hnbian
  * @ description 定义UDAF函数需要实现UserDefinedAggregateFunction中定义的方法
  * @ date 2019/1/15 17:19
  **/
class MapUdaf  extends UserDefinedAggregateFunction {

  //UDAF与DataFrame列有关的输入样式,StructField的名字并没有特别要求，完全可以认为是两个内部结构的列名占位符。
  //至于UDAF具体要操作DataFrame的哪个列，取决于调用者，但前提是数据类型必须符合事先的设置，如这里的DoubleType与DateType类型
  def inputSchema: StructType = {
    StructType(StructField("metric", StringType) :: StructField("timeCategory", StringType) :: StructField("timeCategory", StringType) :: Nil)
  }

  //定义存储聚合运算时产生的中间数据结果的Schema
  def bufferSchema: StructType = {
    StructType(StructField("sumOfCurrent", MapType(StringType, StringType, false)) :: Nil)
  }

  //标明了UDAF函数的返回值
  def dataType: DataType = MapType(StringType, StringType, false)

  //用以标记针对给定的一组输入,UDAF是否总是生成相同的结果
  def deterministic: Boolean = true

  //对聚合运算中间结果的初始化
  def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer.update(0, Map())
  }

  //第二个参数input: Row对应的并非DataFrame的行,而是被inputSchema投影了的行。以本例而言，每一个input就应该只有两个Field的值
  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = Map((input.getString(0), String.format("%s,%s",input.getString(1),input.getString(2))))
  }

  //负责合并两个聚合运算的buffer，再将其存储到MutableAggregationBuffer中
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getMap(0) ++ buffer2.getMap(0)
  }
  def evaluate(buffer: Row): Any = {
    buffer(0)
  }
}
object MapUdaf {
  def main(args: Array[String]): Unit = {
    /*val spark = SparkSession.builder()
      .appName("Map_UDAF")
      .master("local[4]")
      .getOrCreate()
    */


    val spark = SparkUtils.getSparkSession("Map_UDAF",4)
    //如果定义UDAF(User Defined Aggregate Function)
    //Spark为所有的UDAF定义了一个父类UserDefinedAggregateFunction。要继承这个类，需要实现父类的几个抽象方法
    val tag_list = spark.sqlContext.createDataFrame(Seq(
      (1001, "no_1001", "t_1001", "1000", "4"),
      (1001, "no_1001", "t_1002", "2000", "5"),
      (1003, "no_1003", "t_1003", "1000", "2"),
      (1003, "no_1003", "t_1004", "5000", "1"),
      (1005, "no_1005", "t_1005", "4200", "3")))
      .toDF("user_id", "device_id", "tag_id", "watch_count_time", "dt_count")
    tag_list.createTempView("t_tag_list")

    tag_list.show()

    //在使用上，除了需要对UDAF进行实例化之外，与普通的UDF使用没有任何区别
    val map_udaf = new MapUdaf
    spark.sqlContext.udf.register("map_udaf", map_udaf)

    val dataFrame = spark.sqlContext.sql("select user_id,device_id,map_udaf(tag_id,watch_count_time,dt_count) as tag_map from t_tag_list group by  user_id,device_id")
    dataFrame.show(false)
  }
}
