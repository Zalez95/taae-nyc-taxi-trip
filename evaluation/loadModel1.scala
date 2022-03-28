:load common.scala

import org.apache.spark.ml.classification.NaiveBayesModel

var PATH = "./"
var PATH_MODELO = "./"
var FILE = "train.csv"


/*****************************************************************************/
// Guardamos el CSV en un DataFrame
val taxiTripDF = loadCSV(PATH + FILE).persist


/**** LIMPIEZA ****/
val cleanTaxiTripDF = cleanDF(taxiTripDF)


/**** TRANSFORMACION ****/
// Transformar trip_duration a la clase short/long
val shortLongThreshold = 1200
val classTaxiDF = cleanTaxiTripDF.withColumn("trip_duration", when($"trip_duration" < shortLongThreshold.toInt, "short").otherwise("long"))

// Transformar pickup_datetime y dropoff_datetime a pickup_time, pickup_weekday, dropoff_time y dropoff_weekday, de tipo Double y String
var timeTaxiDF = convertDates(classTaxiDF, "pickup_datetime")
timeTaxiDF = convertDates(timeTaxiDF, "dropoff_datetime")

// Longitud y latitud positivas
val longLatTaxiDF = longLatPositive(timeTaxiDF)

// Atributos categoricos a Double, y eliminacion de id
var inputColumns = Array("store_and_fwd_flag", "pickup_datetime_weekday", "dropoff_datetime_weekday")
var outputColumns = inputColumns.map(_ + "_num").toArray
val siColumns= new StringIndexer().setInputCols(inputColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

val taxiSimColumns = siColumns.fit(longLatTaxiDF)
val numericTaxiDF = (taxiSimColumns.transform(longLatTaxiDF)
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
// Cargar el modelo
var taxiFeatLabMd = NaiveBayesModel.load(PATH_MODELO + "modelo")

// Probamos el modelo con los valores actuales
val predictionsAndLabelsDF = taxiFeatLabMd.transform(taxiFeatLabDF).select("prediction", "label")

// TODO: STATS
predictionsAndLabelsDF.show()

// Limpiar
taxiFeatLabDF.unpersist()
taxiTripDF.unpersist()

