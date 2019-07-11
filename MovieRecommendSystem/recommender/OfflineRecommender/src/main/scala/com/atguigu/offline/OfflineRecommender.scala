package com.atguigu.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix


/**
  * Movie数据集，数据集字段通过分割
  *
  * 151^                          电影的ID
  * Rob Roy (1995)^               电影的名称
  * In the highlands ....^        电影的描述
  * 139 minutes^                  电影的时长
  * August 26, 1997^              电影的发行日期
  * 1995^                         电影的拍摄日期
  * English ^                     电影的语言
  * Action|Drama|Romance|War ^    电影的类型
  * Liam Neeson|Jessica Lange...  电影的演员
  * Michael Caton-Jones           电影的导演
  *
  * tag1|tag2|tag3|....           电影的Tag
  **/

case class Movie(val mid: Int, val name: String, val descri: String, val timelong: String, val issue: String,
                 val shoot: String, val language: String, val genres: String, val actors: String, val directors: String)

/**
  * Rating数据集，用户对于电影的评分数据集，用，分割
  *
  * 1,           用户的ID
  * 31,          电影的ID
  * 2.5,         用户对于电影的评分
  * 1260759144   用户对于电影评分的时间
  */
case class MovieRating(val uid: Int, val mid: Int, val score: Double, val timestamp: Int)

/**
  * MongoDB的连接配置
  *
  * @param uri MongoDB的连接
  * @param db  MongoDB要操作数据库
  */
case class MongoConfig(val uri: String, val db: String)

//推荐
case class Recommendation(rid: Int, r: Double)

//用户的推荐
case class UserRecs(uid: Int, recs: Seq[Recommendation])

//电影的相似度
case class MovieRecs(mid: Int, recs: Seq[Recommendation])


object OfflineRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val USER_MAX_RECOMMENDATION = 3
  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"


  //入口方法
  def main(args: Array[String]): Unit = {

    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://172.16.104.13:27017/recommender",
      "mongo.db" -> "reommender"
    )

    //创建一个spark
    val sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores")).set("spark.executor.memory", "2G").set("spark.driver.memory", "2G")

    //创建sparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    //读取MongoDB中的数据
    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))
    import spark.implicits._

    //评分数据
    val ratingRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => (rating.uid, rating.mid, rating.score)).cache()

    //用户数据
    val userRDD = ratingRDD.map(_._1).distinct()

    //电影数据
    val movieRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .rdd
      .map(rating => (rating.mid)).cache()

    //创建训练数据集
    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))

    val (rank, iterations, lambda) = (50, 5, 0.01)

    //训练ALS模型
    val model = ALS.train(trainData, rank, iterations, lambda)

    //计算用户推荐矩阵
    val userMovies = userRDD.cartesian(movieRDD)
    val preRatings = model.predict(userMovies)
    val userRecs = preRatings
      .filter(_.rating > 0)
      .map(rating => (rating.user, (rating.product, rating.rating)))
      .groupByKey()
      .map {
        case (uid, recs) => UserRecs(uid, recs.toList.sortWith(_._2 > _._2).take(USER_MAX_RECOMMENDATION).map(x => Recommendation(x._1, x._2)))
      }.toDF()

    //    userRecs
    //      .write
    //      .option("uri", mongoConfig.uri)
    //      .option("collection", USER_RECS)
    //      .mode("overwrite")
    //      .format("com.mongodb.spark.sql")
    //      .save()

    //计算电影相似度矩阵
    //获取电影的特征矩阵，隐语义
    val movieFeatures = model.productFeatures.map {
      case (mid, features) => (mid, new DoubleMatrix(features))
    }
    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter {
        case (a, b) => a._1 != b._1
      }
      .map {
        case (a, b) => val simScore = this.consinSim(a._2, b._2)
          (a._1, (b._1, simScore))
      }.filter(_._2._2 > 0.8)
      .groupByKey()
      .map {
        case (mid, items) => MovieRecs(mid, items.toList.map(x => Recommendation(x._1, x._2)))
      }.toDF()

    movieRecs
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()



    //关闭spark
    spark.close()


  }

  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
    //l1范数：向量元素绝对值之和；l2范数：即向量的模长（向量的长度）,向量元素的平方和再开方
    //

  }


}
