:load loadCSV.scala
:load dfCleanTransform.scala

import org.apache.spark.ml.classification.DecisionTreeClassificationModel

var PATH = "./"
var PATH_MODELO = "./"
var FILE = "train.csv"


/*****************************************************************************/
/** Guardamos el CSV en un DataFrame */
val taxiTripDF = loadCSV(PATH + FILE).persist

// Limpieza y transformacion
val cleanTaxiTripDF = cleanDF(taxiTripDF)
val taxiFeatLabDF = transformDF(cleanTaxiTripDF).persist


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

