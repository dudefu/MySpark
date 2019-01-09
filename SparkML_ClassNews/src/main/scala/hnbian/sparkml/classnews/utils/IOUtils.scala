package hnbian.sparkml.classnews.utils

import java.io.File

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.util.MLWritable


/**
  * @author hnbian
  * @ description io 工具类
  * @ date 2019/1/7 15:41
  **/
object IOUtils {
  /**
    * 删除指定文件或目录及其子目录和文件
    *
    * @param file 待删除文件/目录路径
    * @return 是否已删除
    */
  def delDir(file: File): Boolean = {
    if (file.isDirectory) {
      val subFileList = file.listFiles()
      for (subFile <- subFileList) {
        delDir(subFile)
      }
      file.delete()
    } else {
      file.delete()
    }
  }
}
