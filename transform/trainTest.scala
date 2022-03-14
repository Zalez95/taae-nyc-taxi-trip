import org.apache.spark.sql.types.{IntegerType, StringType, TimestampType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession, Row, DataFrameStatFunctions}
import org.apache.spark.rdd.{RDD}
import spark.implicits._
import java.text.SimpleDateFormat
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets
import org.apache.spark.classification.DecisionTreeClasifier

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
//Se necesita de entrada labelCol y featuresCol



/**** Parametros del Algoritmo ****/

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

