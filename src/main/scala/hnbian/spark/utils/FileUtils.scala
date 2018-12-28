package hnbian.spark.utils

/**
  * @author hnbian
  *         @ Description 
  *         @ Date 2018/12/28 15:17
  **/
object FileUtils extends App {
 /* val path = this.getFilePath("sample_libsvm_data.txt")
  println(path)*/
  def getFilePath(fileName:String): String ={
    this.getClass().getResource(s"/data/${fileName}").getPath()
  }

}
