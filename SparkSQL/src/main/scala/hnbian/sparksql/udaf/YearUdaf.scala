package hnbian.sparksql.udaf

import java.sql.{Date, Timestamp}

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._


/**
  * @author hnbian
  * @ description 定义UDAF函数,按年聚合后比较,需要实现UserDefinedAggregateFunction中定义的方法
  * @ date 2019/1/15 17:46
  **/
class YearUdaf (current: DateRange) extends UserDefinedAggregateFunction{
  val previous: DateRange = DateRange(subtractOneYear(current.startDate), subtractOneYear(current.endDate))
  println(current)
  println(previous)

  //UDAF与DataFrame列有关的输入样式,StructField的名字并没有特别要求，完全可以认为是两个内部结构的列名占位符。
  //至于UDAF具体要操作DataFrame的哪个列，取决于调用者，但前提是数据类型必须符合事先的设置，如这里的DoubleType与DateType类型
  def inputSchema: StructType = {
    StructType(StructField("metric", DoubleType) :: StructField("timeCategory", DateType) :: StructField("timeCategory", StringType) :: Nil)
  }

  //定义存储聚合运算时产生的中间数据结果的Schema
  def bufferSchema: StructType = {
    StructType(StructField("sumOfCurrent", DoubleType) :: StructField("sumOfPrevious", DoubleType) :: Nil)
  }

  //标明了UDAF函数的返回值类型
  //def dataType: org.apache.spark.sql.types.DataType = DoubleType
  def dataType: DataType = MapType(StringType, StringType, false)

  //用以标记针对给定的一组输入,UDAF是否总是生成相同的结果
  def deterministic: Boolean = true

  //对聚合运算中间结果的初始化
  def initialize(buffer: MutableAggregationBuffer): Unit = {
    println("init method" + buffer.toString() + System.nanoTime())
    buffer.update(0, 0.0)
    buffer.update(1, 0.0)
  }

  //第二个参数input: Row对应的并非DataFrame的行,而是被inputSchema投影了的行。以本例而言，每一个input就应该只有两个Field的值
  def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    println("input=" + input.toString())
    if (current.in(input.getAs[Date](1))) {
      buffer(0) = buffer.getAs[Double](0) + input.getAs[Double](0)
      println("buffer(0)=" + buffer(0))
    }
    if (previous.in(input.getAs[Date](1))) {
      buffer(1) = buffer.getAs[Double](0) + input.getAs[Double](0)
      println("buffer(1)=" + buffer(1))
    }
  }

  //负责合并两个聚合运算的buffer，再将其存储到MutableAggregationBuffer中
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getAs[Double](0) + buffer2.getAs[Double](0)
    buffer1(1) = buffer1.getAs[Double](1) + buffer2.getAs[Double](1)
  }

  //完成对聚合Buffer值的运算,得到最后的结果
  /*def evaluate(buffer: Row): Any = {
    if (buffer.getDouble(1) == 0.0) {
      0.0
    } else {
      (buffer.getDouble(0) - buffer.getDouble(1)) / buffer.getDouble(1) * 100
    }
  }*/

  def evaluate(buffer: Row): Any = {
    /*if (buffer.getDouble(1) == 0.0) {
      "key"+0
    } else {
      (buffer.getDouble(0) - buffer.getDouble(1)) / buffer.getDouble(1) * 100
      //Map("key"->buffer.getDouble(1))
      "key"+buffer.getDouble(1)
    }*/
    Map("key"->buffer.getDouble(1).toString)
  }


  private def subtractOneYear(date: Timestamp): Timestamp = {
    val prev = new Timestamp(date.getTime)
    prev.setYear(prev.getYear - 1)
    prev
  }
}

//定义一个日期范围类
case class DateRange(startDate: Timestamp, endDate: Timestamp) {
  def in(targetDate: Date): Boolean = {
    targetDate.before(endDate) && targetDate.after(startDate)
  }

  override def toString(): String = {
    startDate.toLocaleString() + " " + endDate.toLocaleString();
  }
}

object YearUdaf extends App {

  //val dir = "D:/Program/spark/examples/src/main/resources/"
  //System.setProperty("hadoop.home.dir", "D:\\documents\\GitHub\\winutils\\hadoop-2.6.0")
  System.setProperty("hadoop.home.dir", "D:\\ProgramFiles\\winutils-master\\hadoop-2.7.1")
  import hnbian.spark.utils.SparkUtils
  val spark = SparkUtils.getSparkSession("SparkSession",4)

  /*val spark = SparkSession.builder()
    .appName(s"test")
    //.master(commonPorp.getProperty("master"))
    .master("local[4]")
    //.enableHiveSupport() //使用hive
    .getOrCreate()*/

  //如果定义UDAF(User Defined Aggregate Function)
  //Spark为所有的UDAF定义了一个父类UserDefinedAggregateFunction。要继承这个类，需要实现父类的几个抽象方法
  val salesDF = spark.sqlContext.createDataFrame(Seq(
    (1, "a", 1000.00, 400.00, "AZ", "2014-01-02"),
    (2, "b", 2000.00, 500.00, "CA", "2014-02-01"),
    (3, "c", 1000.00, 200.00, "CA", "2015-01-11"),
    (4, "d", 5000.00, 100.00, "CA", "2015-02-19"),
    (5, "e", 4200.00, 300.00, "MA", "2015-02-18")))
    .toDF("id", "name", "sales", "discount", "state", "saleDate")
  //salesDF.persist().createTempView("sales")
  salesDF.createTempView("sales")
  val current = DateRange(Timestamp.valueOf("2015-01-01 00:00:00"), Timestamp.valueOf("2015-12-31 00:00:00"))

  //在使用上，除了需要对UDAF进行实例化之外，与普通的UDF使用没有任何区别
  val yearOnYear = new YearUdaf(current)
  spark.sqlContext.udf.register("yearOnYear", yearOnYear)

  val dataFrame = spark.sqlContext.sql("select id, yearOnYear(sales, saleDate,discount) as yearOnYear from sales group by id")
  salesDF.printSchema()
  dataFrame.show()
}
