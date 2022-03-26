:load loadCSV.scala
:load dfCleanTransformTT.scala

import org.apache.spark.ml.classification.DecisionTreeClassifier

var PATH = "./"
var PATH_MODELO = "./"
var FILE = "train.csv"


/*****************************************************************************/
/** Guardamos el CSV en un DataFrame */
var taxiTripDF = loadCSV(PATH + FILE).persist


/* Realizamos una partición aleatoria de los datos */
/* 66% para entrenamiento, 34% para prueba */
/* Fijamos seedpara usar la misma partición en distintos ejemplos*/
val taxiTripSplits = taxiTripDF.randomSplit(Array(0.66, 0.34), seed=0)
val trainTaxiTripDF = taxiTripSplits(0)
val testTaxiTripDF = taxiTripSplits(1)


// Limpieza y transformacion
val ttCleanDFArray = cleanDF(trainTaxiTripDF, testTaxiTripDF)
val ttTransformDFArray = transformDF(ttCleanDFArray(0), ttCleanDFArray(1))
val trainTaxiFeatLabDF = ttTransformDFArray(0).persist
val testTaxiFeatLabDF = ttTransformDFArray(1).persist


/**** MODELO ****/
// Calculo de error
def nErrores(df: DataFrame) : Double = {
  df.filter(!(col("prediction").contains(col("label")))).count()
}

def calculoError(df: DataFrame) : Double = {
  nErrores(df) / df.count()
}

// Probamos distintos modelos empleando el conjunto de entrenamiento en proporcion 2/3 y 1/3
def test(df : DataFrame) = {
  val dfSplits = df.randomSplit(Array(0.66, 0.34), seed=0)
  val dfS1 = dfSplits(0).persist
  val dfS2 = dfSplits(1).persist

  def testTree(impureza : String, maxProf : Integer, maxBins : Integer) = {
    val dtc = new DecisionTreeClassifier()
    dtc.setImpurity(impureza)
    dtc.setMaxDepth(maxProf)
    dtc.setMaxBins(maxBins)

    val dfS1Md = dtc.fit(dfS1)
    val predictionsAndLabelsDF = dfS1Md.transform(dfS2).select("prediction", "label")
     
    val error = calculoError(predictionsAndLabelsDF)
    printf("testTree(\"%s\", %d, %d) = %f\n", impureza, maxProf, maxBins, error)
  }

  testTree("gini", 3, 10)
  testTree("gini", 3, 5)
  testTree("gini", 4, 5)
  testTree("gini", 5, 5)
  testTree("gini", 6, 5)
  testTree("gini", 7, 5)
  testTree("gini", 8, 5)
  testTree("gini", 9, 5)
  testTree("gini", 9, 32)
  testTree("gini", 9, 50)

  dfS1.unpersist()
  dfS2.unpersist()
}

test(trainTaxiFeatLabDF)

// Modelo final
val dtTaxiTrip = new DecisionTreeClassifier()
dtTaxiTrip.setImpurity("gini")
dtTaxiTrip.setMaxDepth(9)
dtTaxiTrip.setMaxBins(5)

val trainTaxiFeatLabMd = dtTaxiTrip.fit(trainTaxiFeatLabDF)
val predictionsAndLabelsDF = trainTaxiFeatLabMd.transform(testTaxiFeatLabDF).select("prediction", "label")

var error = calculoError(predictionsAndLabelsDF)
trainTaxiFeatLabMd.toDebugString
printf("Tasa de error = %f\n", error)

// Guardado del modelo final
trainTaxiFeatLabMd.write.overwrite().save(PATH_MODELO + "modelo")

// Limpiar
testTaxiFeatLabDF.unpersist()
trainTaxiFeatLabDF.unpersist()
taxiTripDF.unpersist()

