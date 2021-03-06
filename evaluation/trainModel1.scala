:load common.scala

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

var PATH = "./"
var PATH_MODELO = "./"
var FILE = "train.csv"


/*****************************************************************************/
// Guardamos el CSV en un DataFrame
var taxiTripDF = loadCSV(PATH + FILE).persist


// Realizamos una partición aleatoria de los datos
// 66% para entrenamiento, 34% para prueba
// Fijamos seedpara usar la misma partición en distintos ejemplos
val taxiTripSplits = taxiTripDF.randomSplit(Array(0.66, 0.34), seed=0)
val trainTaxiTripDF = taxiTripSplits(0)
val testTaxiTripDF = taxiTripSplits(1)


/**** LIMPIEZA ****/
val ttCleanDFArray = cleanTTDF(trainTaxiTripDF, testTaxiTripDF)


/**** TRANSFORMACION ****/
// Transformar trip_duration a la clase short/long
val shortLongThreshold = 1200
val classTrainTaxiDF = ttCleanDFArray(0).withColumn("trip_duration", when($"trip_duration" < shortLongThreshold.toInt, "short").otherwise("long"))
val classTestTaxiDF = ttCleanDFArray(1).withColumn("trip_duration", when($"trip_duration" < shortLongThreshold.toInt, "short").otherwise("long"))

// Transformar pickup_datetime y dropoff_datetime a pickup_time, pickup_weekday, dropoff_time y dropoff_weekday, de tipo Double y String
var timeTrainTaxiDF = convertDates(classTrainTaxiDF, "pickup_datetime")
timeTrainTaxiDF = convertDates(timeTrainTaxiDF, "dropoff_datetime")

var timeTestTaxiDF = convertDates(classTestTaxiDF, "pickup_datetime")
timeTestTaxiDF = convertDates(timeTestTaxiDF, "dropoff_datetime")

// Longitud y latitud positivas
val longLatTrainTaxiDF = longLatPositive(timeTrainTaxiDF)
val longLatTestTaxiDF = longLatPositive(timeTestTaxiDF)

// Atributos categoricos a Double, y eliminacion de id
var inputColumns = Array("store_and_fwd_flag", "pickup_datetime_weekday", "dropoff_datetime_weekday")
var outputColumns = inputColumns.map(_ + "_num").toArray
val siColumns= new StringIndexer().setInputCols(inputColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

val trainTaxiSimColumns = siColumns.fit(longLatTrainTaxiDF)
val numericTrainTaxiDF = (trainTaxiSimColumns.transform(longLatTrainTaxiDF)
  .drop(inputColumns:_*)
  .drop("id")
)

val numericTestTaxiDF = (trainTaxiSimColumns.transform(longLatTestTaxiDF)
  .drop(inputColumns:_*)
  .drop("id")
)

// Transformacion 1 de k
val tmpCols = inputColumns
inputColumns = outputColumns
outputColumns = tmpCols.map(_ + "_hot")
val hotColumns = new OneHotEncoder().setInputCols(inputColumns).setOutputCols(outputColumns)

val trainTaxiHotmColumns = hotColumns.fit(numericTrainTaxiDF)
val hotTrainTaxiDF = trainTaxiHotmColumns.transform(numericTrainTaxiDF).drop(inputColumns:_*)

val hotTestTaxiDF = trainTaxiHotmColumns.transform(numericTestTaxiDF).drop(inputColumns:_*)

// Creacion de las columnas features
inputColumns = hotTrainTaxiDF.columns.diff(Array("trip_duration"))
val va = new VectorAssembler().setOutputCol("features").setInputCols(inputColumns)

val trainTaxiFeatClaDF = va.transform(hotTrainTaxiDF).select("features", "trip_duration")
val testTaxiFeatClaDF = va.transform(hotTestTaxiDF).select("features", "trip_duration")

// Creacion de la columna label
val indiceClase = new StringIndexer().setInputCol("trip_duration").setOutputCol("label").setStringOrderType("alphabetDesc")
val trainTaxiFeatLabDF = indiceClase.fit(trainTaxiFeatClaDF).transform(trainTaxiFeatClaDF).drop("trip_duration").persist
val testTaxiFeatLabDF = indiceClase.fit(testTaxiFeatClaDF).transform(testTaxiFeatClaDF).drop("trip_duration").persist


/**** MODELO ****/
// Evaluacion de parametros
val nbTaxiTripEval = new NaiveBayes()
val paramGrid = (new ParamGridBuilder()
  .addGrid(nbTaxiTripEval.modelType, Array("multinomial", "complement"))
  .addGrid(nbTaxiTripEval.smoothing, Array(1.0, 10.0, 100.0))
  .build()
)
val bcEval = (new BinaryClassificationEvaluator()
  .setNumBins(1000)
  .setMetricName("areaUnderROC")
  .setRawPredictionCol("probability")
)
val cv = (new CrossValidator()
  .setEstimator(nbTaxiTripEval)
  .setEvaluator(bcEval)
  .setNumFolds(3)
  .setEstimatorParamMaps(paramGrid)
)

val cvModel = cv.fit(trainTaxiFeatLabDF)
val nbBestModel = cvModel.bestModel.parent.asInstanceOf[NaiveBayes]

val finalModelType = nbBestModel.getModelType
val finalSmoothing = nbBestModel.getSmoothing
printf("Mejor modelType: %s\n", finalModelType)
printf("Mejor smoothing: %f\n", finalSmoothing)


// Modelo final
val nbTaxiTrip = new NaiveBayes()
nbTaxiTrip.setModelType(finalModelType)
nbTaxiTrip.setSmoothing(finalSmoothing)

val trainTaxiFeatLabMd = nbTaxiTrip.fit(trainTaxiFeatLabDF)
val predictionsAndLabelsDF = trainTaxiFeatLabMd.transform(testTaxiFeatLabDF).select("prediction", "label")


// Estadisticas del clasificador
predictionsAndLabelsDF.show()

val predictionsAndLabelsRDD = (predictionsAndLabelsDF
  .select("label", "prediction")
  .rdd.map(r => (r.getAs[Double](0), r.getAs[Double](1)))
)
val bcMetrics = new BinaryClassificationMetrics(predictionsAndLabelsRDD)
val mcMetrics = new MulticlassMetrics(predictionsAndLabelsRDD)

printf("Tasa de error: %f\n", 1.0 - mcMetrics.accuracy)
printf("Matrix de confusion:\n")
mcMetrics.confusionMatrix
printf("Tasa de ciertos positivos: %f\n", mcMetrics.weightedPrecision)
printf("Tasa de falsos positivos: %f\n", mcMetrics.weightedFalsePositiveRate)
printf("Area bajo curva ROC: %f\n", bcMetrics.areaUnderROC)
printf("Curva ROC:\n")
bcMetrics.roc().collect()
printf("Area bajo curva PR: %f\n", bcMetrics.areaUnderPR)

// Guardado del modelo final
trainTaxiFeatLabMd.write.overwrite().save(PATH_MODELO + "modelo1")

// Limpiar
testTaxiFeatLabDF.unpersist()
trainTaxiFeatLabDF.unpersist()
taxiTripDF.unpersist()

