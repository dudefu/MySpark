package com.hnbian.shc

import org.apache.spark.sql.execution.datasources.hbase._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * @author hnbian 2019/4/25 17:24
  *         Apache Spark - Apache HBase连接器 测试代码
  */
object SHC extends App {
  val cat = s"""{
               |"table":{"namespace":"huan", "name":"test", "tableCoder":"PrimitiveType"},
               |"rowkey":"key",
               |"columns":{
               |"rk":{"cf":"rowkey", "col":"key", "type":"string"},
               |"deviceNo":{"cf":"info", "col":"deviceNo", "type":"string"}
               |}
               |}""".stripMargin

  System.setProperty("hadoop.home.dir", "D:\\ProgramFiles\\winutils-master\\hadoop-2.7.1")
  val spark = SparkSession.builder()
    .appName("HBaseSourceExample")
    .master("local[4]")
    .getOrCreate()

  val sc = spark.sparkContext
  val sqlContext = spark.sqlContext

  import sqlContext.implicits._

  def withCatalog(cat: String): DataFrame = {
    sqlContext
      .read
      .options(Map(HBaseTableCatalog.tableCatalog->cat))
      .format("org.apache.spark.sql.execution.datasources.hbase")
      .load()
  }

  val df = withCatalog(cat)
  df.show(false)
}
