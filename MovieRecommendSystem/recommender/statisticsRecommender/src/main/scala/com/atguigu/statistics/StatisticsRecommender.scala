package com.atguigu.statistics

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession


//统计推荐算法
//统计所有历史数据中每个电影的评分数
//统计以月为单位拟每个电影的评分数
//统计每个电影的平均评分
//统计每种电影类型中评分最高的10个电影

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
case class Rating(val uid: Int, val mid: Int, val score: Double, val timestamp: Int)


/**
  * MongoDB的连接配置
  *
  * @param uri MongoDB的连接
  * @param db  MongoDB要操作数据库
  */
case class MongoConfig(val uri: String, val db: String)


/**
  *
  * @param rid 推荐movie的mid
  * @param r   movie的评分
  */
case class Recommendation(rid: Int, r: Double)

/**
  *
  * @param genres 电影的类别
  * @param recs   top10的电影的合集
  */
case class GenresRecommendation(genres: String, recs: Seq[Recommendation])

object StatisticsRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"

  val RATE_MORE_COLLECTION = "RateMoreMovies"
  val RATE_MORE_RECENTLY_MOVIES = "RateMoreRecentlyMovies"
  val AVERAGE_MOVIES = "AverageMovies"
  val GENRES_TOP_MOVIES = "GenresTopMovies"

  //入口方法
  def main(args: Array[String]): Unit = {

    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://172.16.104.13:27017/recommender",
      "mongo.db" -> "recommender"
    )


    //创建sparkconf
    val sparkConf = new SparkConf().setAppName("StatisticsRecommender").setMaster(config("spark.cores"))

    //创建sparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))


    import spark.implicits._

    //将数据加载
    val ratingDF = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Rating]
      .toDF()

    val movieDF = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .toDF()

    //将ratingDF创建为一张表以便sql查数据
    ratingDF.createOrReplaceTempView("ratings")

    //第一个需求：统计所有历史数据中每个电影的评分数
    //最终结果数据结构：mid,count

    val rateMoreMoviesDF = spark.sql("select mid,count(mid) as count from ratings group by mid")

    rateMoreMoviesDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", RATE_MORE_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //第二个需求：统计以月为单位拟每个电影的评分数
    //最终结果数据结构：mid,count,time

    //创建一个日期格式化工具
    val simpleDateFormat = new SimpleDateFormat("yyyyMM")

    //注册一个UDF函数，用于将timestamp装换成年月格式   1260759144000  => 201605
    spark.udf.register("changeDate", (x: Int) => simpleDateFormat.format(new Date(x * 1000L)).toInt)

    //将原来的Rating数据集中的时间转换成年月的格式
    val ratingOfYearMonth = spark.sql("select mid,score,changeDate(timestamp) as yearmonth from ratings") //就是上边DF转化来的那张表

    //将新的数据集注册成为一张表
    ratingOfYearMonth.createOrReplaceTempView("ratingOfMonth")

    val rateMoreRecentlyMovies = spark.sql("select mid,count(mid) as count ,yearmonth from ratingOfMonth group by yearmonth,mid")

    rateMoreRecentlyMovies
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", RATE_MORE_RECENTLY_MOVIES)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //第三个需求：统计每个电影的平均评分
    val averageMoviesDF = spark.sql("select mid,avg(score) as avg from ratings group by mid")

    averageMoviesDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", AVERAGE_MOVIES)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //第四个需求：统计每种电影类型中评分最高的10个电影
    val movieWithScore = movieDF.join(averageMoviesDF, Seq("mid", "mid"))

    //所欲的电影类别
    val genres = List("Action", "Adventure", "Animation", "Comedy", "Ccrime", "Documentary", "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery"
      , "Romance", "Science", "Tv", "Thriller", "War", "Western")

    //将电影类别转换成RDD
    val genresRDD = spark.sparkContext.makeRDD(genres)

    //计算电影类别TOP10
    val genresTopMovies = genresRDD.cartesian(movieWithScore.rdd) //电影类别与有评分的电影的笛卡尔积
      .filter {
      case (genres, row) => row.getAs[String]("genres").toLowerCase.contains(genres.toLowerCase)
    }
      .map {

        case (genres, row) => {
          (genres, (row.getAs[Int]("mid"), row.getAs[Double]("avg")))
        }
      }.groupByKey()
      .map {
        case (genres, items) => GenresRecommendation(genres, items.toList.sortWith(_._2 > _._2).take(10).map(item => Recommendation(item._1, item._2)))

      }.toDF()

    //输出数据到mongod

    genresTopMovies
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", GENRES_TOP_MOVIES)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //关闭spark
    spark.stop()

  }

}
