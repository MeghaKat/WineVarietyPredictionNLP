// Databricks notebook source
//Data Preprocessing
//Importing the dataset
val df=spark.table("winemag_data_clean").selectExpr("country","description","cast(points as int) points","cast(price as int) price","province","region","variety","title")

//Dropping duplicate rows and rows with empty values
val df_NA=df.na.drop()
val df_no_na=df_NA.dropDuplicates()
val df_final=df_no_na.dropDuplicates("description", "variety")
df_final.count()

// COMMAND ----------

//Data Exploration: Getting no.of descriptions per varetiy and sorting in descending order
import org.apache.spark.sql.functions._
val df_varities=df_final.groupBy("variety").agg(count("variety")).orderBy(desc("count(variety)")).limit(10)
df_varities.show()

// COMMAND ----------

//Data Preprocessing Filtering out Top 10 Varieties
import org.apache.spark.sql.expressions.Window

val filtered_df =df_final.filter(df_final("variety").isin("pinot noir", "chardonnay", "cabernet sauvignon", "syrah", "malbec", "ros", "tempranillo", "nebbiolo", "sauvignon blanc", "zinfandel"))
filtered_df.select("variety").distinct().count()

// COMMAND ----------

//Splitting the data 70/30
val seed = 100
val weights = Array(0.7,0.3)
val splitDF =filtered_df.randomSplit(weights, seed)
val (train_df,test_df) = (splitDF(0),splitDF(1))


// COMMAND ----------

print("train_df",train_df.count())
print("test_df",test_df.count())

// COMMAND ----------

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer,NGram}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.Column
//Data Preprocessing 
val tokenizer = new Tokenizer().setInputCol("cleaned_description").setOutputCol("vector")

// #Remove stop words
val remover = new StopWordsRemover().setInputCol("vector").setOutputCol("vector_no_stopwords")


//Cleaning the Data
def clean_text(col: Column): Column = {
  var t=lower(col)
  t=regexp_replace(col, "\\s+", "")
  t=regexp_replace(col, "[^a-zA-Z0-9\\s]+","")
  t=regexp_replace(col, "[\\d]","")
   return t
  
}
//Generating N-grams
val ngram = new NGram().setN(1).setInputCol("vector_no_stopwords").setOutputCol("unigram")

def preprocess_text(input_df:DataFrame): DataFrame={
 val  clean_df = input_df.withColumn("cleaned_description",clean_text(col("description")))
  val tokenized_df = tokenizer.transform(clean_df)
  val no_stop_words_df = remover.transform(tokenized_df)
 var output_df = ngram.transform(no_stop_words_df)
  output_df = output_df.where(size(col("unigram")) >0)
  return output_df
}                    
                                         
val train_production_df = preprocess_text(train_df)
val test_production_df = preprocess_text(test_df)


// COMMAND ----------

//Print a single row after preprocessing
test_production_df.select("unigram").show(1,false)

// COMMAND ----------

import org.apache.spark.ml.classification.{LogisticRegression,DecisionTreeClassifier,RandomForestClassifier,NaiveBayes,GBTClassifier,OneVsRest}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel,StringIndexer,HashingTF, IDF, Tokenizer,VectorAssembler,OneHotEncoderEstimator,PCA,Word2Vec}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StandardScaler
//=========================Tried but, dint work======================================================
//Hashing
// val hashingTF = new HashingTF().setInputCol("unigram").setOutputCol("rawFeatures").setNumFeatures(300)
//IDF
// val idf = new IDF().setInputCol("rawFeatures").setOutputCol("tfidf")
//PCA
// val pca = new PCA().setInputCol("vec_features").setOutputCol("features").setK(10)
//Word2Vec
// val word2Vec = new Word2Vec().setInputCol("unigram").setOutputCol("word_vec").setVectorSize(3).setMinCount(0)
//Scaling
// val scaler = new StandardScaler().setInputCol("features_assembler").setOutputCol("features").setWithStd(true).setWithMean(false)

//========================================================================================================
//CountVectorizer
val count_vectorizer = new CountVectorizer().setInputCol("unigram").setOutputCol("count_vec").setVocabSize(200).setMinDF(5)

//StringIndexing
val country_indexer = new StringIndexer().setInputCol("country").setOutputCol("country_idx").setHandleInvalid("keep")
val variety_indexer = new StringIndexer().setInputCol("variety").setOutputCol("label").setHandleInvalid("keep")
val title_indexer = new StringIndexer().setInputCol("title").setOutputCol("title_idx").setHandleInvalid("keep")
val region_indexer = new StringIndexer().setInputCol("region").setOutputCol("region_idx").setHandleInvalid("keep")

//OneHotEncoding
val encoder = new OneHotEncoderEstimator().setInputCols(Array("country_idx","title_idx","region_idx","label")).setOutputCols(Array("country_enc","title_enc","region_enc","variety_enc"))

//Vector Assembler 
val vectorAssembler = new VectorAssembler().setInputCols(Array("country_enc","title_enc","region_enc","points","price","count_vec")).setOutputCol("features")

//Pipeline
val pipeline = new Pipeline().setStages(Array(count_vectorizer,country_indexer,variety_indexer,title_indexer,region_indexer,encoder, vectorAssembler))
val model = pipeline.fit(train_production_df)
val train_final_df_p=model.transform(train_production_df)
val test_final_df_p=model.transform(test_production_df)

// COMMAND ----------

//Determining the variety labels
import org.apache.spark.ml.feature.{StringIndexerModel}
val stringIndexerModel = model.stages(2).asInstanceOf[StringIndexerModel]
val variety_labels = stringIndexerModel.labels
print("variety_labels: ", variety_labels)

// COMMAND ----------

// Saving to DBFS for SVD-Singular-Value Decomposition i.e. Dimensionality reduction
train_final_df_p.write.saveAsTable("train_vector_assembled")
test_final_df_p.write.saveAsTable("test_vector_assembled")


//GOTO: Python Notebook: 'SVD PySpark Final' for Dimensionality Reduction

// COMMAND ----------

//Loading dataframe with features and label after SVD
val train_dt=spark.table("train_svd_600_n").select("pca_features","label").withColumnRenamed("pca_features", "features")
val test_dt=spark.table("test_svd_600_n").select("pca_features","label").withColumnRenamed("pca_features", "features")


// COMMAND ----------

train_dt.show()

// COMMAND ----------

//XGBoost
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
val xgbParam = Map("eta" -> 0.2f,
      "missing" -> -999,
      "objective" -> "multi:softprob",
      "num_class" -> 11,
      "num_round" -> 100,
      "num_workers" -> 2,
      "colsample_bytree"-> 0.6,
      "subsample"-> 0.7)

val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")
xgbClassifier.setMaxDepth(8)
val xgbClassificationModel = xgbClassifier.fit(train_dt)
val train_final_df = xgbClassificationModel.transform(train_dt)
val test_final_df = xgbClassificationModel.transform(test_dt)

// clf = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)

// COMMAND ----------

//XGBoost hyperparameters
import org.apache.spark.ml.tuning._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
val evaluator=new MulticlassClassificationEvaluator()
val paramGrid = new ParamGridBuilder()
    .addGrid(xgbClassifier.maxDepth, Array(3, 8))
    .addGrid(xgbClassifier.eta, Array(0.1,0.2, 0.6))
    .build()
val cv = new CrossValidator()
     .setEstimator(xgbClassifier)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)

val cvModel = cv.fit(train_dt)
val bestModel = cvModel.bestModel.asInstanceOf[XGBoostClassificationModel]
bestModel.extractParamMap()

// COMMAND ----------

//SVC and logistic 
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression,DecisionTreeClassifier,RandomForestClassifier,NaiveBayes,GBTClassifier,OneVsRest}
//Logistic
val lr = new LogisticRegression().setMaxIter(100).setRegParam(0.01).setFitIntercept(true)
//GBT-Gradient Booster Tree Classifier
val gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").setMaxIter(100)
//Support Vector Classifier
val lsvc = new LinearSVC().setMaxIter(500).setRegParam(0.001)
//One Vs Rest Classifier
//Replace with gbt or lsvc as per usage
val ovr = new OneVsRest().setClassifier(lsvc)

val pipeline1 = new Pipeline().setStages(Array(ovr))
val model1 = pipeline1.fit(train_dt)
val train_final_df=model1.transform(train_dt)
val test_final_df=model1.transform(test_dt)

// COMMAND ----------

import org.apache.spark.ml.classification.{OneVsRestModel,LinearSVCModel}
var map=model1.stages(0).asInstanceOf[OneVsRestModel].extractParamMap().classifier


// COMMAND ----------

//Metric Evaluation for any model
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val train_evaluator= new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
val train_accuracy = train_evaluator.setMetricName("accuracy").evaluate(train_final_df)
val F1 = train_evaluator.setMetricName("f1").evaluate(train_final_df)
val precision = train_evaluator.setMetricName("weightedPrecision").evaluate(train_final_df)
val recall = train_evaluator.setMetricName("weightedRecall").evaluate(train_final_df)


println("=============================================================================================")

val test_evaluator= new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
val test_accuracy = train_evaluator.setMetricName("accuracy").evaluate(test_final_df)
val test_F1 = train_evaluator.setMetricName("f1").evaluate(test_final_df)
val test_precision = train_evaluator.setMetricName("weightedPrecision").evaluate(test_final_df)
val test_recall = train_evaluator.setMetricName("weightedRecall").evaluate(test_final_df)


// COMMAND ----------

//CrossValidation Logistic--Hyperparameter Tuning
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
val paramGrid = new ParamGridBuilder()
  .addGrid(lr1.regParam, Array(0.1, 0.01,0.001,1))
  .addGrid(lr1.maxIter, Array(50, 100,200,400))
  .addGrid(lr1.fitIntercept, Array(true,false))
  .build()
val cv = new CrossValidator()
  .setEstimator(pipeline1)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)
  .setParallelism(2)  
val cvModel = cv.fit(train_dt)
val cvModel_test=cvModel.transform(test_dt)



val test_evaluator_cv = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val test_accuracy_cv = test_evaluator.evaluate(cvModel_test)

test_evaluator_cv.extractParamMap()

// COMMAND ----------

//CrossValidation SVC--Hyperparameter Tuning
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
val paramGrid = new ParamGridBuilder()
.addGrid(ovr. eta, Array(0.1, 0.01,0.001,1))
  	.addGrid(ovr. colsample_bytree, Array(50, 100,200,400, 500))
 	 .addGrid(ovr. subsample, Array(true,false))
.addGrid(ovr. num_round, Array(true,false))
.addGrid(ovr. MaxDepth=8, Array(true,false))
  .build()
val cv = new CrossValidator()
  .setEstimator(pipeline1)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)
  .setParallelism(2)  
val cvModel = cv.fit(train_dt)
val cvModel_test=cvModel.transform(test_dt)

val test_evaluator_cv = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val test_accuracy_cv = test_evaluator.evaluate(cvModel_test)

test_evaluator_cv.extractParamMap()

// COMMAND ----------

//SVC and logistic --- Set Best Model Parameters


import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression,DecisionTreeClassifier,RandomForestClassifier,NaiveBayes,GBTClassifier,OneVsRest}
//Logistic
val lr = new LogisticRegression().setMaxIter(100).setRegParam(0.01).setFitIntercept(true)
//Support Vector Classifier
val lsvc = new LinearSVC().setMaxIter(500).setRegParam(0.001)
//One Vs Rest Classifier
val ovr = new OneVsRest().setClassifier(lsvc)

//Pass ovr for SVC or lr for Logistic Regression
val pipeline1 = new Pipeline().setStages(Array(ovr))

val model1 = pipeline1.fit(train_dt)
val train_final_df=model1.transform(train_dt)
val test_final_df=model1.transform(test_dt)

// COMMAND ----------

//Saving the best Model results for Confusion Matrix --- change table name as required
train_final_df.write.saveAsTable("train_xgboost_450F_results_n")
test_final_df.write.saveAsTable("test_xgboost_450F_results_n")

//GOTO: Python Notebook  'Confusion Matrix'

// COMMAND ----------


