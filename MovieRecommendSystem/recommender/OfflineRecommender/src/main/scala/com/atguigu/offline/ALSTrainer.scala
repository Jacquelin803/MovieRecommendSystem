package com.atguigu.offline

import breeze.numerics.sqrt
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object ALSTrainer {
  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://172.16.104.13:27017/recommender",
      "mongo.db" -> "recommender"
    )

    //创建SparkConf
    val sparkConf = new SparkConf().setAppName("ALSTrainer").setMaster(config("spark.cores"))

    //创建SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val mongoConfig = MongoConfig(config("mongo.uri"),config("mongo.db"))

    import spark.implicits._
    //加载评分数据
    val ratingRDD=spark
      .read
      .option("uri",mongoConfig.uri)
      .option("collection",OfflineRecommender.MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating=>Rating(rating.uid,rating.mid,rating.score)).cache()
          //一定要加cache(),否则会溢出

    //输出最优参数
    adjustALSParams(ratingRDD)

    //关闭spark
    spark.close()



  }
  //输出最优参数函数
  def adjustALSParams(trainData:RDD[Rating]):Unit={
    val result=for (rank<-Array(30,40,50,70,80);lambda<-Array(1,0.1,0.001))
      yield {
        val model=ALS.train(trainData,rank,5,lambda)
        val rmse=getRmse(model,trainData)
        (rank,lambda,rmse)
        //for循环中的 yield 会把当前的元素记下来，保存在集合中，循环结束后将返回该集合。
      }
    println(result.sortBy(_._3).head)      //从小到大
  }
  def getRmse(model: MatrixFactorizationModel, trainData: RDD[Rating]):Double={
    //构造一个userproducts RDD[(Int,Int)]
    val userMovies=trainData.map(item=>(item.user,item.product))
    val predictRating=model.predict(userMovies)
    val real=trainData.map(item=>((item.user,item.product),item.rating))
    val predict=predictRating.map(item=>((item.user,item.product),item.rating))
      //RMSE公式https://blog.csdn.net/capecape/article/details/78623897
    sqrt(
      real.join(predict).map{
        case ((uid,mid),(real,pre))=>
          val err=real-pre
          err*err
      }.mean()
    )
  }

}
