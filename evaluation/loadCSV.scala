import org.apache.spark.sql.{DataFrame, SparkSession}
import java.nio.file.{Paths, Files}
import org.apache.spark.sql.types.{IntegerType, StringType, TimestampType, DoubleType, StructField, StructType}
import java.nio.charset.StandardCharsets


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

