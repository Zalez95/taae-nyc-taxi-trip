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
import org.apache.spark.ml.classification.DecisionTreeClassifier

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
//Transformar pickup_datetime y dropoff_datetime a pickup_time y dropoff_time, de tipo Double
def convertDates(df: DataFrame, column: String) : DataFrame = {
  var df2 = df
  var Array(name, _*) = column.split("_")
     df2 = df2.withColumn(name+"_hour", date_format(col(column), "H"))
     df2 = df2.withColumn(name+"_minute", date_format(col(column), "m"))
     df2 = df2.withColumn(name+"_second", date_format(col(column), "s"))
     df2 = df2.withColumn(name+"_time", col(name+"_hour")*3600 + col(name+"_minute")*60 + col(name+"_second"))
     df2 = df2.drop(name+"_hour")
     df2 = df2.drop(name+"_minute")
     df2 = df2.drop(name+"_second")
     df2 = df2.drop(column)
     return df2
}

//Transformar pickup_datetime y dropoff_datetime a pickup_time y dropoff_time, de tipo Double
taxiTripDF = convertDates(taxiTripDF, "pickup_datetime")
taxiTripDF = convertDates(taxiTripDF, "dropoff_datetime")

// Transformar trip_duration a la clase short/long
val medianTripDuration = hotTrainTaxiDF.stat.approxQuantile("trip_duration", Array(0.5), 0.0001)(0)
val classTrainTaxiDF = cleanTrainTaxiTripDF.withColumn("trip_duration", when($"trip_duration" < medianTripDuration.toInt, "short").otherwise("long"))
val classTestTaxiDF = cleanTestTaxiTripDF.withColumn("trip_duration", when($"trip_duration" < medianTripDuration.toInt, "short").otherwise("long"))

// Atributos categoricos a Double, y eliminacion de id
var inputColumns = classTrainTaxiDF.columns.filter(_ == "store_and_fwd_flag").toArray
var outputColumns = inputColumns.map(_ + "_num").toArray
val siColumns= new StringIndexer().setInputCols(inputColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

val trainTaxiSimColumns = siColumns.fit(classTrainTaxiDF)
val numericTrainTaxiDF = (trainTaxiSimColumns.transform(classTrainTaxiDF)
  .drop(inputColumns:_*)
  .drop("id")
)

val numericTestTaxiDF = (trainTaxiSimColumns.transform(classTestTaxiDF)
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

// Se necesita de entrada labelCol y featuresCol


/**** MODELO ****/

//crear la instancia del modelo
val DTtrip=new DecisionTreeClassifier()

//Parametros del modelo

val impureza = "gini" //Se selecciona gini por tener atributos numericos
val maxProf = 3
val maxBins = 5 //es el default, mirar si va bien, porque es critico

//fijar parametros del modelo
DTtrip.setImpurity(impureza)
DTtrip.setMaxDepth(maxProf)
DTtrip.setMaxBins(maxBins)

//taxiTrip.unpersist()

