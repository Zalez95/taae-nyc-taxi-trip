import org.apache.spark.sql.types.{IntegerType, StringType, TimestampType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession, Row, DataFrameStatFunctions}
import org.apache.spark.rdd.{RDD}
import spark.implicits._
import java.text.SimpleDateFormat
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets

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


def myStatsDouble(attribute:String) = {
  var rdd = (taxiTripDF.select(attribute)
          .rdd.map(r => { r.getDouble(0) })
          .persist
  )

  println(rdd.stats)

  val hist = rdd.histogram(40)
  sc.parallelize(hist._2, 1).saveAsTextFile(attribute + "Values")
  sc.parallelize(hist._1, 1).saveAsTextFile(attribute + "Ranges")

  rdd.unpersist()
}

def myStatsInt(attribute:String) = {
  var rdd = (taxiTripDF.select(attribute)
          .rdd.map(r => { r.getInt(0) })
          .persist
  )

  println(rdd.stats)

  var df2 = (taxiTripDF.select(attribute)
    .groupBy(attribute).count
    .sort(attribute)
    .persist
  )
  sc.parallelize(df2.select("count").rdd.map(r => { r.getLong(0) }).collect, 1)
    .saveAsTextFile(attribute + "Values")
  sc.parallelize(df2.select(attribute).rdd.map(r => { r.getInt(0) }).collect, 1)
    .saveAsTextFile(attribute + "Tags")

  df2.unpersist()
  rdd.unpersist()
}

def myStatsString(attribute:String) = {
  taxiTripDF.groupBy(attribute).count.show
}

def myStatsTimestamp(attribute:String) = {
  val dateformat = new SimpleDateFormat("yyyy-MM-dd")
  val df2 = (taxiTripDF.select(attribute)
    .map(r => { dateformat.format(r.getTimestamp(0)) })
    .groupBy("value").count
    .sort("value")
    .persist
  )

  sc.parallelize(df2.select("count").rdd.map(r => { r.getLong(0) }).collect, 1)
    .saveAsTextFile(attribute + "Values")
  sc.parallelize(df2.select("value").rdd.map(r => { r.getString(0) }).collect, 1)
    .saveAsTextFile(attribute + "Tags")

  df2.unpersist()
}

def myStatsCoords(latAttr:String, lonAttr:String) = {
  val precission = 100.0
  val rdd = (taxiTripDF.select(latAttr, lonAttr)
    .map(r => {
      val lat = r.getDouble(0)
      val lon = r.getDouble(1)
      ((precission * lat).round / precission, (precission * lon).round / precission)
    })
    .groupBy("_1", "_2").count
    .sort(desc("count"))
    .rdd
  )

  rdd.map(_.toString().replace("[","").replace("]", ""))
    .saveAsTextFile(latAttr + lonAttr)
}


println("id")
myStatsString("id")
println("vendor_id")
myStatsInt("vendor_id")
println("pickup_datetime")
myStatsTimestamp("pickup_datetime")
println("dropoff_datetime")
myStatsTimestamp("dropoff_datetime")
println("passenger_count")
myStatsInt("passenger_count")
println("pickup_longitude")
myStatsDouble("pickup_longitude")
println("pickup_latitude")
myStatsDouble("pickup_latitude")
println("dropoff_longitude")
myStatsDouble("dropoff_longitude")
println("dropoff_latitude")
myStatsDouble("dropoff_latitude")
println("store_and_fwd_flag")
myStatsString("store_and_fwd_flag")
println("trip_duration")
myStatsInt("trip_duration")


// Correlations
println("Correlations")

val safFlagDF = (taxiTripDF
  .select("id", "store_and_fwd_flag")
  .rdd.map(r => {
    if (r.getString(1) == "Y")
      (r.getString(0), 1)
    else
      (r.getString(0), 0)
  })
  .toDF("id2", "store_and_fwd_flag_int")
)
val taxiTripDF2 = taxiTripDF.join(safFlagDF, taxiTripDF("id") === safFlagDF("id2")).persist

val columns = Seq(
  "vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude",
  "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag_int"
)

var cols = ""
var corrs = ""
columns.foreach((column1: String) => {
  var corrs2 = ""
  columns.foreach((column2: String) => {
    val corr = taxiTripDF2.stat.corr(column1, column2)
    if (corrs2 == "")
      corrs2 = corr.toString
    else
      corrs2 = corrs2 + "," + corr.toString
  })

  if (cols == "") {
    cols = column1
    corrs = corrs2
  }
  else {
    cols = cols + "," + column1
    corrs = corrs + "\n" + corrs2
  }
})

Files.write(Paths.get("correlacionesValues.txt"), corrs.getBytes(StandardCharsets.UTF_8))
Files.write(Paths.get("correlacionesTags.txt"), cols.getBytes(StandardCharsets.UTF_8))


taxiTripDF2.unpersist()
taxiTripDF.unpersist()

