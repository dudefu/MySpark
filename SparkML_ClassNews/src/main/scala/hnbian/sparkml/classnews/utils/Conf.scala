package hnbian.sparkml.classnews.utils

import java.io.{File, FileInputStream, InputStreamReader}
import java.util.Properties

import scala.collection.mutable


/**
  * @author hnbian
  *         @ Description
  *         @ Date 2019/1/7 14:51
  **/
object Conf extends Serializable {
  /**
    * 加载配置文件
    *
    * @param filePath 配置文件路径
    * @return Map存储的配置参数
    */
  def loadConf(filePath: String): mutable.LinkedHashMap[String, String] = {
    val kvMap = mutable.LinkedHashMap[String, String]()

    val properties = new Properties()
    properties.load(new InputStreamReader(Conf.getClass.getClassLoader.getResourceAsStream(filePath), "UTF-8"))

    val propertyNameArray = properties.stringPropertyNames().toArray(new Array[String](0))
    val fileName = new File(filePath).getName
    println(s"============ 加载配置文件 $fileName ================")
    for (propertyName <- propertyNameArray) {
      val property = properties.getProperty(propertyName).replaceAll("\"", "").trim
      println(propertyName + ": " + property)
      kvMap.put(propertyName, property)
    }
    println("==========================================================")

    kvMap
  }
}
