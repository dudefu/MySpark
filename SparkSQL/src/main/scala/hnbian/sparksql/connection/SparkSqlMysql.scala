package hnbian.sparksql.connection

import hnbian.spark.utils.SparkUtils

/**
  * @author hnbian 2019/1/15 17:44
  * */
object SparkSqlMysql extends App {

  val spark = SparkUtils.getSparkSession("SparkSqlMysql", 4)

  val prop = new java.util.Properties
  prop.put("user", "root")
  prop.put("password", "root")
  prop.put("driver", "com.mysql.jdbc.Driver")
  val url = "jdbc:mysql://slave1:3306/spark"
  val table = "users"
  //注册驱动
  Class.forName("com.mysql.jdbc.Driver")
  val df = spark.read.jdbc(url, table, prop)
  //读取数据
  df.show
  /*df.foreach(e=>{
    println(e.getLong(0) + " : " + e.getString(1)+" : "+ e.getString(2))
  })*/
  //写入数据

  df.filter("userid=3").write.jdbc(url, "spark03", prop)
}
