// Databricks notebook source
/*loading the datasets
Movie information is in the file "movies.dat" and is in the following
format:

MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including
year of release)
- Genres are pipe-separated and are selected from the following genres:

	* Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western
*/

import org.apache.spark.storage.StorageLevel
val movies = sc.textFile("/FileStore/tables/movies.dat").map(_.split("::"))
movies.take(10)

// COMMAND ----------

/*
All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

User information is in the file "users.dat" and is in the following
format:

UserID::Gender::Age::Occupation::Zip-code

All demographic information is provided voluntarily by the users and is
not checked for accuracy.  Only users who have provided some demographic
information are included in this data set.

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"
*/


val ratings = sc.textFile("/FileStore/tables/ratings.dat").map(_.split("::"))
val users = sc.textFile("/FileStore/tables/users.dat").map(_.split("::"))

// COMMAND ----------

/*Calculating the animated movies with ratings 4 or above

We take out all those movie_id's from the movies dataset which are having 'Animation' as their genre
We take out the details of the movies from ratings dataset that have been rated >= 4
Then we join these two results based on the movie_id to get the 'Animated' movies that have been rated >= 4

*/


case class moviesClass(movie_id:Int, title:String, genre:String)
val movies_DF = movies.map(row => moviesClass(row(0).trim.toInt, row(1), row(2))).toDF()
movies_DF.createOrReplaceTempView("movies_table")
movies_DF.show()
val animation_movies_id = spark.sql("select movie_id from movies_table where genre like '%Animation%'")//Each movie belongs to one or more genres so taking all movies that belong to Animation genre
animation_movies_id.show()
//animation_movies_id.createOrReplaceTempView("abc")

// COMMAND ----------

case class ratingsClass(user_id:Int, movie_id:Int, rating:Int, timestamp:Long)
val ratings_DF = ratings.map(row => ratingsClass(row(0).trim.toInt, row(1).trim.toInt, row(2).trim.toInt, row(3).trim.toInt)).toDF()
ratings_DF.createOrReplaceTempView("ratings_table")
ratings_DF.show()
val ratings_greater_than_4 = spark.sql("select * from ratings_table where rating >= 4")
ratings_greater_than_4.show()
//ratings_greater_than_4.createOrReplaceTempView("pqr")
ratings_greater_than_4.count()

// COMMAND ----------

val animated_movies_greater_than_4 = animation_movies_id.as("a").join(ratings_greater_than_4.as("b"), $"a.movie_id" === $"b.movie_id")//joining the two DF's based on movie_id to get all animated movies rated >= 4
animated_movies_greater_than_4.show()
animated_movies_greater_than_4.printSchema()

// COMMAND ----------

//finding the gender bias for each genre of movies
case class userClass(user_Id:Int, gender:String, age:Int, zip_code:Long)
val user_DF = users.map(row => userClass(row(0).trim.toInt, row(1), row(2).trim.toInt, row(3).trim.toLong)).toDF()

// COMMAND ----------

user_DF.createOrReplaceTempView("userTable")
user_DF.printSchema()
ratings_DF.printSchema()
movies_DF.printSchema()

// COMMAND ----------

val gender_bias_on_movie_id = spark.sql("select count(movie_id), genre from movies_table group by genre")
gender_bias_on_movie_id.show()

// COMMAND ----------

/*

grouping the ratings by age
extract the user_id and rating from ratings data
extract user_id and age from user data
join them based on user_id

*/
val user_rating = spark.sql("select user_id as users_id, rating from ratings_table")
user_rating.show()
user_rating.count()

// COMMAND ----------

val user_age = spark.sql("select user_id as id, age from userTable")
user_age.show()
user_age.count()

// COMMAND ----------

/*

size of user_rating DF = 1000209
size of user_age DF = 6040
As we see the size of one DF is much larger than the size of the other. So here if we use a shuffle join operation then we would have some serious performance issues. So instead we use a broadast join where the smaller DF is sent as a broadcast variable to all the the executors of the larger DF. This increases the performance of the join.

*/
import org.apache.spark.sql.functions._
val ratings_by_age = user_rating.as("a").join(broadcast(user_age).as("b")).where($"a.users_id" === $"b.id")
ratings_by_age.show()


// COMMAND ----------

ratings_by_age.createOrReplaceTempView("ratings")
//finding the no of ratings that were received for each age groups
spark.sql("select count(rating) as no_of_rating, age from ratings group by age order by age").show()

// COMMAND ----------

//finding the average rating for each movies from the ratings data

val avg_rating_of_movies = spark.sql("select avg(rating) as avg_rating, movie_id from ratings_table group by movie_id order by movie_id")
avg_rating_of_movies.show()

// COMMAND ----------

ratings_DF.printSchema()
movies_DF.printSchema()

// COMMAND ----------

/*

finding the titles of the best rated movies
extract the movie_id and title from movie data
Join it with the average ratings calculated in previous question based on movie_id

*/
val title_DF = spark.sql("select movie_id as id, title from movies_table")
val avg_rating_DF = avg_rating_of_movies.as("a").join(title_DF.as("b")).where($"a.movie_id" === $"b.id")
avg_rating_DF.show()
avg_rating_DF.count()

// COMMAND ----------

avg_rating_DF.createOrReplaceTempView("best_titles_table")
val best_titles_DF = spark.sql("select distinct(movie_id), title, avg_rating from best_titles_table order by avg_rating desc")//displaying the top 20 movies with the highest average ratings based on the ratings given by the users
best_titles_DF.show()
best_titles_DF.count()

// COMMAND ----------

/*
Now we want to use the ratings given by the users in order to design a movie recommendation system for the users
This model will generate top 10 movie recommendation for each user based on the historical data which contains the ratings given by the users for movies that they have watched previously.

In turn the model will also predict the top 10 users who are likely to watch a particular movie so that we can target those users.

For building a recommendation engine we would the collaborative filtering method of Machine Learning.
The ALS(Alternating Least Squares) approach of MlLib implements collaborative filtering to predict the missing entries in the user-item association matrix.

//Designing the recomendation system**
*/

//importing the MlLib packages into our code.
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

// COMMAND ----------

/*
***********************************************
exploring the nature of the ratings data and then randomly dividing the entire dataset into training and test data for applying to our model.
Out of the total data
Training data --> 80%
Test data --> 20%
*********************************************
*/
ratings_DF.printSchema()

//dividing the dataset randomly into training and test set
val Array(training, test) = ratings_DF.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

/*Building the ALS model
Define the ALS class with the parameters

************************************
No. of iterations = 10
Regularization Parameter = 0.01
users Column in the dataset = "user_id"
Item/Product/movie column = "movie_id"
User Ratings column = "rating"
***********************************

Fit the model using the training data
*/

val als = new ALS().setMaxIter(10).setRegParam(0.01).setUserCol("user_id").setItemCol("movie_id").setRatingCol("rating")
val model = als.fit(training)//creating the model using the training data

// COMMAND ----------

/*dropping all those rows having NaN
Spark allows us to set the coldStartStrategy parameter to drop in order to drop all those rows from the predictions DataFrame that contain NaN-values.
Then we can evaluate the model on the non-NaN data and will be valid.

*/
model.setColdStartStrategy("drop")//setting coldStartStrategy() to drop means we drop all NaN rows from the dataframe
//Evaluating the predictions on the test data
val predictions = model.transform(test)
predictions.printSchema()

// COMMAND ----------

/*
Evaluating the model by calculating the Root Mean Square Error of the predictions.
We define the RegressionEvaluator() class with metric name as rmse, label column as "rating" and prediction column as "prediction"

*/
val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"The Root Mean Squared Error = $rmse")

// COMMAND ----------

/*
Here we recommend 10 movie_id's for each user_id along with the corresponding rating for each movie and store it as "movie_1.json"

*/
//top 10 movie recommendations for each user
val userRecs = model.recommendForAllUsers(10)
userRecs.show()
userRecs.printSchema()
userRecs.write.json("/FileStore/tables/movie_1.json")//storing the recommendations as json format file.

// COMMAND ----------

/*
Here we recommend top 10 user_id's for each movie_id along with the corresponding average rating given by each user and store it as "users_1.json"

*/
//top 10 user recommendation for each movies
val movieRecs = model.recommendForAllItems(10)
movieRecs.show()
movieRecs.printSchema()
movieRecs.write.json("/FileStore/tables/users_1.json")//store the user recommendations in json format.

// COMMAND ----------

/*
Recomending movies for a set of 3 users *********
*/
//Recommend movies for a specified set of users
val users = ratings_DF.select(als.getUserCol).distinct().limit(3)
val userSubsetRatings = model.recommendForUserSubset(users, 10)
userSubsetRatings.show()

// COMMAND ----------

/*
Recomending users for a set of 3 movies *********
*/
//Recommend Users for specified subset of movies
val movies = movies_DF.select(als.getItemCol).distinct.limit(3)
val userSubsetRatings = model.recommendForItemSubset(movies, 10)
userSubsetRatings.show()

// COMMAND ----------


