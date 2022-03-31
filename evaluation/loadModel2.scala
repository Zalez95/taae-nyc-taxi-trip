:load common.scala

import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

var PATH = "./"
var PATH_MODELO = "./modelo"
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
// Cargar el modelo
var taxiFeatLabMd = RandomForestClassificationModel.load(PATH_MODELO)

// Probamos el modelo con los valores actuales
val predictionsAndLabelsDF = taxiFeatLabMd.transform(taxiFeatLabDF).select("prediction", "label")

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

// Limpiar
taxiFeatLabDF.unpersist()
taxiTripDF.unpersist()

