import org.apache.spark.sql.types.{IntegerType, StringType, TimestampType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession, Row, DataFrameStatFunctions}
import org.apache.spark.rdd.{RDD}
import spark.implicits._
import java.text.SimpleDateFormat
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

var PATH = "./"
var FILE = "train.csv"


/*****************************************************************************/
var taxiTripSchema = StructType(Array(
	StructField("id", StringType, true),
	StructField("vendor_id", IntegerType, true),
	StructField("pickup_datetime", TimestampType, true),
	StructField("dropoff_datetime", TimestampType, true),
	StructField("passenger_count", IntegerType, true),
	StructField("pickup_longitude", DoubleType, true),
	StructField("pickup_latitude", DoubleType, true),
	StructField("dropoff_longitude", DoubleType, true),
	StructField("dropoff_latitude", DoubleType, true),
	StructField("store_and_fwd_flag", StringType, true),
	StructField("trip_duration", IntegerType, true)
));

var taxiTripDF = (spark.read
        .option("header", "true")
	.option("delimiter", ",")
        .format("csv")
	.schema(taxiTripSchema).load(PATH + FILE)
	.persist
)


/* Realizamos una partición aleatoria de los datos */
/* 66% para entrenamiento, 34% para prueba */
/* Fijamos seedpara usar la misma partición en distintos ejemplos*/
val taxiTripSplits = taxiTripDF.randomSplit(Array(0.66, 0.34), seed=0)
val trainTaxiTripDF = taxiTripSplits(0)
val testTaxiTripDF = taxiTripSplits(1)


/**** LIMPIEZA ****/
// No hay valores nulos, pero por si acaso eliminamos valores nulos
var cleanTrainTaxiTripDF = trainTaxiTripDF.na.drop()
var cleanTestTaxiTripDF = testTaxiTripDF.na.drop()
val ntaie = testTaxiTripDF.count() - cleanTestTaxiTripDF.count()

// Outliers - Eliminar datos del dia 23 de enero
def filterDates(df : DataFrame) : DataFrame = {
  df.filter($"pickup_datetime" < "2016-01-23" || $"pickup_datetime" > "2016-01-24")
    .filter($"dropoff_datetime" < "2016-01-23" || $"dropoff_datetime" > "2016-01-24")
    .filter($"dropoff_datetime" < "2016-07-01")
}

var ntoe = cleanTestTaxiTripDF.count()
cleanTrainTaxiTripDF = filterDates(cleanTrainTaxiTripDF)
cleanTestTaxiTripDF = filterDates(cleanTestTaxiTripDF)
ntoe = ntoe - cleanTestTaxiTripDF.count()

val tasaNoClasificados = (ntaie + ntoe).toDouble / testTaxiTripDF.count().toDouble


/**** TRANSFORMACION ****/
def convertDates(df: DataFrame, column: String) : DataFrame = {
  df.withColumn(column + "_time", date_format(col(column), "H")*3600 + date_format(col(column), "m")*60 + date_format(col(column), "s"))
    .withColumn(column + "_weekday", dayofweek(col(column)).cast(StringType))
    .drop(column)
}

// Transformar trip_duration a la clase short/long
val medianTripDuration = cleanTrainTaxiTripDF.stat.approxQuantile("trip_duration", Array(0.5), 0.0001)(0)
val classTrainTaxiDF = cleanTrainTaxiTripDF.withColumn("trip_duration", when($"trip_duration" < medianTripDuration.toInt, "short").otherwise("long"))
val classTestTaxiDF = cleanTestTaxiTripDF.withColumn("trip_duration", when($"trip_duration" < medianTripDuration.toInt, "short").otherwise("long"))

// Transformar pickup_datetime y dropoff_datetime a pickup_time, pickup_weekday, dropoff_time y dropoff_weekday, de tipo Double y String
var timeTrainTaxiDF = convertDates(classTrainTaxiDF, "pickup_datetime")
timeTrainTaxiDF = convertDates(timeTrainTaxiDF, "dropoff_datetime")

var timeTestTaxiDF = convertDates(classTestTaxiDF, "pickup_datetime")
timeTestTaxiDF = convertDates(timeTestTaxiDF, "dropoff_datetime")

// Atributos categoricos a Double, y eliminacion de id
var inputColumns = Array("store_and_fwd_flag", "pickup_datetime_weekday", "dropoff_datetime_weekday")
var outputColumns = inputColumns.map(_ + "_num").toArray
val siColumns= new StringIndexer().setInputCols(inputColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

val trainTaxiSimColumns = siColumns.fit(timeTrainTaxiDF)
val numericTrainTaxiDF = (trainTaxiSimColumns.transform(timeTrainTaxiDF)
  .drop(inputColumns:_*)
  .drop("id")
)

val numericTestTaxiDF = (trainTaxiSimColumns.transform(timeTestTaxiDF)
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
val trainTaxiFeatLabDF = indiceClase.fit(trainTaxiFeatClaDF).transform(trainTaxiFeatClaDF).drop("trip_duration")
val testTaxiFeatLabDF = indiceClase.fit(testTaxiFeatClaDF).transform(testTaxiFeatClaDF).drop("trip_duration")


/**** MODELO ****/
// crear la instancia del modelo
val dtTaxiTrip = new DecisionTreeClassifier()

// Parametros del modelo
val impureza = "gini" // Se selecciona gini por tener atributos numericos
val maxProf = 3
val maxBins = 5       // es el default, mirar si va bien, porque es critico

dtTaxiTrip.setImpurity(impureza)
dtTaxiTrip.setMaxDepth(maxProf)
dtTaxiTrip.setMaxBins(maxBins)

val trainTaxiFeatLabMd = dtTaxiTrip.fit(trainTaxiFeatLabDF)
trainTaxiFeatLabMd.toDebugString

// Predecimos la clase de los ejemplos de prueba
val predictionsAndLabelsDF = trainTaxiFeatLabMd.transform(testTaxiFeatLabDF).select("prediction", "label")

// Calculamos estadisticas de la prediccion
val metrics = new MulticlassClassificationEvaluator()
metrics.setMetricName("accuracy")

val acierto = metrics.evaluate(predictionsAndLabelsDF)
val error = 1.0 - acierto

println(f"Tasa de error = $error%1.3f")


dtTaxiTrip.unpersist()

