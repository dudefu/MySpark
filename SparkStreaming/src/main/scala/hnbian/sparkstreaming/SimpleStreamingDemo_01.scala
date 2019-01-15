package hnbian.sparkstreaming

/**
  * yum install netcat
  * nc -lk 9999
  * Created by admin on 2018/2/11.
  * 简单的Streamingdemo
  */
object SimpleStreamingDemo_01 extends  App{
  import org.apache.spark._
  //创建一个 local StreamingContext , 包含两个工作线程,并将批次讲个设为1秒
  //master至少需要两个CPU核,以避免出现任务饿死的情况
  val appName = "streaming"
  val master = "local[4]"
  val conf = new SparkConf().setAppName(appName).setMaster(master)
  val ssc = new StreamingContext(conf, Seconds(5))

  // 创建一个连接到hostname:port的DStream，如：localhost:9999
  val lines = ssc.socketTextStream("192.168.1.232", 9999)

  // 将每一行分割成多个单词
  val words = lines.flatMap(_.split(" ")) // Spark 1.3之后不再需要这行
  // 对每一批次中的单词进行计数
  val pairs = words.map(word => (word, 1))
  val wordCounts = pairs.reduceByKey(_ + _)
  // 将该DStream产生的RDD的头十个元素打印到控制台上
  wordCounts.print()

  /*// 每隔10秒归约一次最近30秒的数据
  val windowedWordCounts = pairs.reduceByKeyAndWindow((a:Int,b:Int) => (a + b), Seconds(30), Seconds(10))
  */
  ssc.start()             // 启动流式计算
  ssc.awaitTermination()  // 等待直到计算终止


}
