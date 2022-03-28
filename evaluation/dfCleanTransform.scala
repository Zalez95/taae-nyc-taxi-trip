import org.apache.spark.sql.{DataFrame, DataFrameStatFunctions}
import org.apache.spark.sql.types.{IntegerType, StringType, TimestampType, DoubleType, StructField, StructType}
import spark.implicits._
import java.text.SimpleDateFormat
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoderModel
import org.apache.spark.ml.feature.VectorAssembler


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


/** TRANSFORMACION */
def transformDF(taxiTripDF : DataFrame) : DataFrame = {
  def convertDates(df: DataFrame, column: String) : DataFrame = {
    df.withColumn(column + "_time", date_format(col(column), "H")*3600 + date_format(col(column), "m")*60 + date_format(col(column), "s"))
      .withColumn(column + "_weekday", dayofweek(col(column)).cast(StringType))
      .drop(column)
  }

  // Transformar trip_duration a la clase short/long
  val shortLongThreshold = 1200
  val classTaxiDF = taxiTripDF.withColumn("trip_duration", when($"trip_duration" < shortLongThreshold.toInt, "short").otherwise("long"))

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
  indiceClase.fit(taxiFeatClaDF).transform(taxiFeatClaDF).drop("trip_duration")
}

