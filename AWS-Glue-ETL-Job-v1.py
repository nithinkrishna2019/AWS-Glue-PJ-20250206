import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col,trim,lit

## @params: [JOB_NAME]
args=getResolvedOptions(sys.argv, ['JOB_NAME'])#when running gluejob there are many params passsed ,here in this line of code we are extracting 'JOB_NAME' and storing it in 'args' , "JOB_NAME" Identifies the Glue job as a whole.

#sys.argv --> List of arguments passed when the job runs.
#'JOB_NAME' --> The name of the argument we need to extract.
#'getResolvedOptions()'-->Extracts only the required argument from sys.argv.
#Why is it needed? --> AWS Glue requires a job name to track execution.

sc = SparkContext() #SparkContext is the entry point for working with Apache Spark. (Tells Spark to start running.) , its like engine starter in a car

#Allocates resources memory & CPUs for data processing.
#In AWS Glue, it is needed to enable distributed data processing . (Allows interaction with Spark RDDs (Resilient Distributed Datasets))


glueContext = GlueContext(sc) #Extend Spark with Glue features

spark = glueContext.spark_session #Allows to work with data using dataframes and sql.


job = Job(glueContext) #This creates a Glue job object to manage the job execution.

#It tracks the job’s state (start, progress, success, failure)
#It is required when you want to commit job progress in AWS Glue.


job.init(args['JOB_NAME'], args) #Initializes the job with a name.


glue_dynamic_frame_initial = glueContext.create_dynamic_frame.from_catalog(database='my-aws-glue-db', table_name='ufo_reports_source_csv_table')

#You want to use DynamicFrame when Data that does not confirm to a fixed schema.
#Data is stored in AWS Glue’s optimized format. AWS Glue’s special version of a DataFrame, optimized for ETL.
#Can be converted to Spark DataFrame

#what happense if we dont use DynamicFrame ??
#1.) We cannot directly load AWS Glue Catalog tables 2.) Glue’s built-in transformations wont work 3.)Schema evolution and automatic data cleaning won’t be available.

df_spark = glue_dynamic_frame_initial.toDF() # DynamicFrame to Spark df

## Rename the columns, extract year and drop unnecessary columns. Remove NULL records"""

from pyspark.sql.functions import col, trim
import pyspark.sql.functions as F

def prepare_dataframe(df):
    """Rename columns, extract year, drop unnecessary columns, and remove rows containing NULLs or blanks."""
    
    # Rename columns for consistency
    df_renamed = df.withColumnRenamed("Shape Reported", "shape_reported") \
        .withColumnRenamed("Colors Reported", "color_reported") \
        .withColumnRenamed("State", "state") \
        .withColumnRenamed("Time", "time")

    # Extract year and drop unnecessary columns
    df_year_added = df_renamed.withColumn("year", F.year(F.to_timestamp(F.col("time"), "M/d/yyyy H:mm"))) \
        .drop("time") \
        .drop("city")

    # Explicitly filter out NULLs and blank ("") values in shape_reported and color_reported columns
    df_final = df_year_added.filter(
        (col("shape_reported").isNotNull()) & (trim(col("shape_reported")) != "") &
        (col("color_reported").isNotNull()) & (trim(col("color_reported")) != "")
    )

    return df_final

#Create color and shape dataframes and join them"

def join_dataframes(df):
    shape_grouped = df.groupBy("year", "state", "shape_reported") \
        .agg(F.count("*").alias("shape_occurrence"))

    color_grouped = df.groupBy("year", "state", "color_reported") \
        .agg(F.count("*").alias("color_occurrence"))

    df_joined = shape_grouped.join(color_grouped,
                                on=["year", "state"],
                                how="inner")
    
    return df_joined

def create_final_dataframe(df):
    """Create final dataframe"""
    shape_window_spec = Window.partitionBy("year", "state").orderBy(F.col("shape_occurrence").desc()) 
    color_window_spec = Window.partitionBy("year", "state").orderBy(F.col("color_occurrence").desc())

    # Selecting top occurrences of shape and color per year and state
    final_df = df.withColumn("shape_rank", F.row_number().over(shape_window_spec)) \
        .withColumn("color_rank", F.row_number().over(color_window_spec)) \
        .filter((F.col("shape_rank") == 1) & (F.col("color_rank") == 1)) \
        .select("year", "state", "shape_reported", "shape_occurrence", "color_reported", "color_occurrence") \
        .orderBy(F.col("shape_occurrence").desc())
    
    return final_df

df_prepared = prepare_dataframe(df_spark)
df_joined = join_dataframes(df_prepared)
df_final = create_final_dataframe(df_joined)

# From Spark dataframe to glue dynamic frame
glue_dynamic_frame_final = DynamicFrame.fromDF(df_final, glueContext, "glue_etl") #	"glue_etl" Labels and tracks a specific transformation step within the job.

# Write the data in the DynamicFrame to a location in Amazon S3 and a table for it in the AWS Glue Data Catalog
s3output = glueContext.getSink(
  path="s3://aws-glue-s3-bucket/ufo_reports_target_parquet",
  connection_type="s3",
  updateBehavior="UPDATE_IN_DATABASE",
  partitionKeys=[],
  compression="snappy",
  enableUpdateCatalog=True,
  transformation_ctx="s3output",
)

s3output.setCatalogInfo(
  catalogDatabase="my-aws-glue-db", catalogTableName="ufo_reports_destination_table"
)

s3output.setFormat("glueparquet") # writes data in paraquet format and s3output knows where to send the data 
s3output.writeFrame(glue_dynamic_frame_final) #triggers the writing of your transformed data to the specified destination



## job.commit() should be the last line of teh script
job.commit() # Without job.commit(), AWS Glue doesn’t know the job is finished!

