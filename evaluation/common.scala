import org.apache.spark.sql.{DataFrame, SparkSession, DataFrameStatFunctions}
import java.nio.file.{Paths, Files}
import org.apache.spark.sql.types.{IntegerType, StringType, TimestampType, DoubleType, StructField, StructType}
import java.nio.charset.StandardCharsets
import spark.implicits._
import java.text.SimpleDateFormat
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.feature.VectorAssembler


/** LOAD CSV */
def loadCSV(path : String) : DataFrame = {
  val taxiTripSchema = StructType(Array(
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

  spark.read
    .option("header", "true")
    .option("delimiter", ",")
    .format("csv")
    .schema(taxiTripSchema).load(path)
}


/** LIMPIEZA */
def cleanDF(taxiTripDF : DataFrame) : DataFrame = {
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

  cleanTaxiTripDF
}


/** LIMPIEZA */
def cleanTTDF(trainTaxiTripDF : DataFrame, testTaxiTripDF : DataFrame) : Array[DataFrame] = {
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
  printf("Tasa de no clasificados: %f\n", tasaNoClasificados)

  Array(cleanTrainTaxiTripDF, cleanTestTaxiTripDF)
}


/** TRANFORMACION FECHAS */
def convertDates(df: DataFrame, column: String) : DataFrame = {
  df.withColumn(column + "_time", date_format(col(column), "H")*3600 + date_format(col(column), "m")*60 + date_format(col(column), "s"))
    .withColumn(column + "_weekday", dayofweek(col(column)).cast(StringType))
    .drop(column)
}


/** TRANFORMACION LATITUD Y LONGITUD */
def longLatPositive(df : DataFrame) : DataFrame = {
  df.withColumn("pickup_longitude", col("pickup_longitude") + 180)
    .withColumn("dropoff_longitude", col("dropoff_longitude") + 180)
    .withColumn("pickup_latitude", col("pickup_latitude") + 90)
    .withColumn("dropoff_latitude", col("dropoff_latitude") + 90)
}

