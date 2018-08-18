import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS


/*
         We want to use the ratings given by the users in order to design a movie recommendation system for the users
        This model will generate top 10 movie recommendation for each user based on the historical data which contains the ratings given by the users for movies that they have watched previously.

        In turn the model will also predict the top 10 users who are likely to watch a particular movie so that we can target those users.

        For building a recommendation engine we would the collaborative filtering method of Machine Learning.
        The ALS(Alternating Least Squares) approach of MlLib implements collaborative filtering to predict the missing entries in the user-item association matrix.

        //Designing the recomendation system**
        */


object RecommendationEngine{

    case class RatingsClass(user_id:Int, movie_id:Int, rating:Int, timestamp:Long)

    def main(args: Array[String]){

        val conf = new SparkConf().setAppName("Recommendation Engine")
        val sc = new SparkContext(conf) // ** Spark Context created and is now available as sc **
        /*
        ** Now lets create the spark session object from the spark context we just now created **
        */
        val spark = SparkSession.builder()
                           .master("local")
                           .appName("Recommendation Engine")
                           .getOrCreate()

        import spark.implicits._

        val ratings = sc.textFile("/FileStore/tables/ratings.dat").map(_.split("::"))
        //reading the data from textfile into RDD
        //case class RatingsClass(user_id:Int, movie_id:Int, rating:Int, timestamp:Long)
        val ratings_DF = ratings.map(row => RatingsClass(row(0).trim.toInt, row(1).trim.toInt, row(2).trim.toInt, row(3).trim.toInt)).toDF()
        ratings_DF.createOrReplaceTempView("ratings_table")
        ratings_DF.createOrReplaceTempView("ratings_table")

        /*
        ***********************************************
        exploring the nature of the ratings data and then randomly dividing the entire dataset into training and test data for applying to our model.
        Out of the total data
        Training data --> 80%
        Test data --> 20%
        *********************************************
        */


        val Array(training, test) = ratings_DF.randomSplit(Array(0.8, 0.2))
        val als = new ALS().setMaxIter(10).setRegParam(0.01).setUserCol("user_id").setItemCol("movie_id").setRatingCol("rating")
        val model = als.fit(training)//creating the model using the training data

        model.setColdStartStrategy("drop")//setting coldStartStrategy() to drop means we drop all NaN rows from the dataframe
        //Evaluating the predictions on the test data
        val predictions = model.transform(test)

        /*
        Evaluating the model by calculating the Root Mean Square Error of the predictions.
        We define the RegressionEvaluator() class with metric name as rmse, label column as "rating" and prediction column as "prediction"
        */
        val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
        val rmse = evaluator.evaluate(predictions)
        println(s"The Root Mean Squared Error = $rmse")

        /*
        Here we recommend 10 movie_id's for each user_id along with the corresponding rating for each movie and store it as "movie_1.json"
        */
        //top 10 movie recommendations for each user
        val userRecs = model.recommendForAllUsers(10)
        userRecs.show()
        userRecs.printSchema()
        userRecs.write.json("/FileStore/tables/movie_1.json")//storing the recommendations as json format file.

        /*
        Here we recommend top 10 user_id's for each movie_id along with the corresponding average rating given by each user and store it as "users_1.json"
        */
        //top 10 user recommendation for each movies
        val movieRecs = model.recommendForAllItems(10)
        movieRecs.show()
        movieRecs.printSchema()
        movieRecs.write.json("/FileStore/tables/users_1.json")//store the user recommendations in json format.



    }

}



