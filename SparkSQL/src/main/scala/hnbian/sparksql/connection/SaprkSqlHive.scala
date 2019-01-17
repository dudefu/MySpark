package hnbian.sparksql.connection

import java.sql.DriverManager

/**
  * @author hnbian 2019/1/15 17:43
  **/
object SaprkSqlHive extends App {
  //注册驱动
  Class.forName("org.apache.hive.jdbc.HiveDriver")
  //val conn = DriverManager.getConnection("jdbc:hive2://master1:10000")
  val conn = DriverManager.getConnection("jdbc:hive2://master:10000")
  val st = conn.createStatement()
  //val rs = st.executeQuery("show databases")
  val rs = st.executeQuery("select count(1) from tableName")
  while(rs.next()){
    println(rs.getString(1))
  }
  println("over! . . .")
}
