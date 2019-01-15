package hnbian.sparkcore.sharedvariables

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.Map

/**
  * Created by admin on 2017/4/6.
  * 需要注册，不然在运行过程中，会抛出一个序列化异常。
  * val accumulator = new My2AccumulatorV2()
  *sc.register(accumulator)
  */
class MyAccumulatorV2 extends AccumulatorV2[String, String] {

  var map = Map[String, Int]()
  var result = ""

  //当AccumulatorV2中存在类似数据不存在这种问题时，是否结束程序。
  override def isZero: Boolean = {
    true
  }

  //拷贝一个新的AccumulatorV2
  override def copy(): AccumulatorV2[String, String] = {
    val myAccumulator = new MyAccumulatorV2()
    myAccumulator.result = this.result
    myAccumulator
  }

  override def reset(): Unit = {
    result = ""
  }

  override def add(word: String): Unit = {

    if (map.getOrElse(word, 0) == 0) {
      map += (word -> 1)
    } else {
      map.update(word, map.get(word).get + 1)
    }
    result = map.toString()
    result
  }

  override def merge(other: AccumulatorV2[String, String]) = other match {
    case map: MyAccumulatorV2 =>
      result = other.value
    case _ =>
      throw new UnsupportedOperationException(
        s"Cannot merge ${this.getClass.getName} with ${other.getClass.getName}")
  }

  override def value: String = {
    result
  }
}
