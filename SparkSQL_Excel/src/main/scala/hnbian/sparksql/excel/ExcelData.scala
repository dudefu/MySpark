package hnbian.sparksql.excel

import hnbian.spark.utils.SparkUtils
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._

/**
  * @author hnbian 2019/3/5 14:29
  */
object ExcelData {
  def main(args: Array[String]): Unit = {
    val spark = SparkUtils.getSparkSession("ExcelData", 4)
    val sqlContext = spark.sqlContext

    val filePath = "D:\\test.xlsx"
    val fileSavePath = "D:\\testWrite.xlsx"

    //定义数据结构
    val schema = StructType(List(
      StructField("c1", StringType, nullable = false),
      StructField("c2", StringType, nullable = false),
      StructField("c3", StringType, nullable = false),
      StructField("c4", StringType, nullable = false),
      StructField("c5", StringType, nullable = false),
      StructField("c6", DateType, nullable = false)))

    val df = load(filePath,spark,schema)//读取Excel 文件
    save(fileSavePath,df) //把刚刚都出来的内容写到另外一个文件中（复制上一个文件）
  }

  /**
    * 将数据保存到Excel文件中
    * @param filePath 保存路径
    * @param df 数据集
    */
  def save(filePath:String,df:DataFrame): Unit ={
    df.write
      .format("com.crealytics.spark.excel")
      .option("dataAddress", "'Sheet'!A1:E2")
      .option("useHeader", "true")
      //.option("dateFormat", "yy-mmm-d") // Optional, default: yy-m-d h:mm
      //.option("timestampFormat", "mm-dd-yyyy hh:mm:ss") // Optional, default: yyyy-mm-dd hh:mm:ss.000
      .mode("append") // Optional, default: overwrite.
      .save(filePath)

  }

  /**
    * 加载Excel数据
    * @param filePath 文件路今天
    * @param spark SparkSession
    * @param schema 数据结构
    */
  def load(filePath:String,spark:SparkSession,schema:StructType): DataFrame ={
    val df = spark.read
      .format("com.crealytics.spark.excel")
      .option("dataAddress", "'Sheet2'!A1:E2") // 可选,设置选择数据区域 例如 A1:E2。
      .option("useHeader", "false") // 必须，是否使用表头，false的话自己命名表头（_c0）,true则第一行为表头
      .option("treatEmptyValuesAsNulls", "true") // 可选, 是否将空的单元格设置为null ,如果不设置为null 遇见空单元格会报错 默认t: true
      .option("inferSchema", "true") // 可选, default: false
      //.option("addColorColumns", "true") // 可选, default: false
      //.option("timestampFormat", "yyyy-mm-dd hh:mm:ss") // 可选, default: yyyy-mm-dd hh:mm:ss[.fffffffff]
      //.option("excerptSize", 6) // 可选, default: 10. If set and if schema inferred, number of rows to infer schema from
      //.option("workbookPassword", "pass") // 可选, default None. Requires unlimited strength JCE for older JVMs====
      //.option("maxRowsInMemory", 20) // 可选, default None. If set, uses a streaming reader which can help with big files====
      .schema(schema) // 可选, default: Either inferred schema, or all columns are Strings
      .load(filePath)
    df.show()
    df
  }
}
