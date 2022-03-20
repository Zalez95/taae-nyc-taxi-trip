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
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

var PATH = "./"
var PATH_MODELO = "./"
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

/**** LIMPIEZA ****/
// Eliminamos valores nulos
var cleanTaxiTripDF = taxiTripDF.na.drop()
val ntaie = taxiTripDF.count() - cleanTaxiTripDF.count()

// Outliers - Eliminar datos del dia 23 de enero
def filterDates(df : DataFrame) : DataFrame = {
  df.filter($"pickup_datetime" < "2016-01-23" || $"pickup_datetime" > "2016-01-24")
    .filter($"dropoff_datetime" < "2016-01-23" || $"dropoff_datetime" > "2016-01-24")
    .filter($"dropoff_datetime" < "2016-07-01")
}

var ntoe = cleanTaxiTripDF.count()
cleanTaxiTripDF = filterDates(cleanTaxiTripDF)
ntoe = ntoe - cleanTaxiTripDF.count()

val tasaNoClasificados = (ntaie + ntoe).toDouble / taxiTripDF.count().toDouble
printf("Tasa de no clasificados = %f\n", tasaNoClasificados)


/**** TRANSFORMACION ****/
def convertDates(df: DataFrame, column: String) : DataFrame = {
  df.withColumn(column + "_time", date_format(col(column), "H")*3600 + date_format(col(column), "m")*60 + date_format(col(column), "s"))
    .withColumn(column + "_weekday", dayofweek(col(column)).cast(StringType))
    .drop(column)
}

// Transformar trip_duration a la clase short/long
val medianTripDuration = 663.0  // Fijado al valor que tenia el conjunto de entrenamiento para evitar que la clase generada sea distinta
val classTaxiDF = cleanTaxiTripDF.withColumn("trip_duration", when($"trip_duration" < medianTripDuration.toInt, "short").otherwise("long"))

// Transformar pickup_datetime y dropoff_datetime a pickup_time, pickup_weekday, dropoff_time y dropoff_weekday, de tipo Double y String
var timeTaxiDF = convertDates(classTaxiDF, "pickup_datetime")
timeTaxiDF = convertDates(timeTaxiDF, "dropoff_datetime")

// Atributos categoricos a Double, y eliminacion de id
var inputColumns = Array("store_and_fwd_flag", "pickup_datetime_weekday", "dropoff_datetime_weekday")
var outputColumns = inputColumns.map(_ + "_num").toArray
val siColumns= new StringIndexer().setInputCols(inputColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

val taxiSimColumns = siColumns.fit(timeTaxiDF)
val numericTaxiDF = (taxiSimColumns.transform(timeTaxiDF)
  .drop(inputColumns:_*)
  .drop("id")
)

// Transformacion 1 de k
val tmpCols = inputColumns
inputColumns = outputColumns
outputColumns = tmpCols.map(_ + "_hot")
val hotColumns = new OneHotEncoder().setInputCols(inputColumns).setOutputCols(outputColumns)

val taxiHotmColumns = hotColumns.fit(numericTaxiDF)
val hotTaxiDF = taxiHotmColumns.transform(numericTaxiDF).drop(inputColumns:_*)

// Creacion de las columnas features
inputColumns = hotTaxiDF.columns.diff(Array("trip_duration"))
val va = new VectorAssembler().setOutputCol("features").setInputCols(inputColumns)

val taxiFeatClaDF = va.transform(hotTaxiDF).select("features", "trip_duration")

// Creacion de la columna label
val indiceClase = new StringIndexer().setInputCol("trip_duration").setOutputCol("label").setStringOrderType("alphabetDesc")
val taxiFeatLabDF = indiceClase.fit(taxiFeatClaDF).transform(taxiFeatClaDF).drop("trip_duration").persist


/**** MODELO ****/
// Calculo de error
def nErrores(df: DataFrame) : Double = {
  df.filter(!(col("prediction").contains(col("label")))).count()
}

def calculoError(df: DataFrame) : Double = {
  nErrores(df) / df.count()
}

// Cargar el modelo
var taxiFeatLabMd = DecisionTreeClassificationModel.load(PATH_MODELO + "modelo")

// Probamos el modelo con los valores actuales
val predictionsAndLabelsDF = taxiFeatLabMd.transform(taxiFeatLabDF).select("prediction", "label")

var error = calculoError(predictionsAndLabelsDF)
printf("Tasa de error = %f\n", error)

// Limpiar
taxiFeatLabDF.unpersist()
taxiTripDF.unpersist()

