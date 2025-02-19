import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import coalesce,col,when,sum,avg,row_number,count,year,month,dayofmonth,to_date,date_format,to_timestamp,trim,rank,row_number,dense_rank
from pyspark.sql.window import Window
from datetime import datetime

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

timestamp_backup = datetime.now().strftime("%Y%m%d_%H%M%S")
s3_path = f"s3://aws-glue-s3-bucket/ufo_reports_source_csv_backup/{timestamp_backup}/"

df_spark.coalesce(1).write.mode("overwrite").csv(s3_path, header=True) #taking backup before processing further

## Rename the columns, extract year and drop unnecessary columns. Remove NULL records"""

def prepare_dataframe(df):
    """Rename columns, extract year, drop unnecessary columns, and remove rows containing NULLs or blanks."""
    
    # Rename columns for consistency
    df_renamed = df.withColumnRenamed("Shape Reported", "shape_reported") \
        .withColumnRenamed("Colors Reported", "color_reported") \
        .withColumnRenamed("State", "state") \
        .withColumnRenamed("Time", "time")
    
    df_renamed = df_renamed.filter(
        (col("shape_reported").isNotNull()) & (trim(col("shape_reported")) != "") &
        (col("color_reported").isNotNull()) & (trim(col("color_reported")) != ""))
    
    timestamp_col = coalesce(
        to_timestamp(col("time"), "M/d/yyyy H:mm"),
        to_timestamp(col("time"), "MM-dd-yyyy HH:mm")
    )

    df_year_added = df_renamed.withColumn("year", year(timestamp_col)).drop("city").drop("time")

    return df_year_added

#Create color and shape dataframes and join them"

def join_dataframes(df):
    shape_grouped = df.groupBy("year", "state", "shape_reported") \
        .agg(count("*").alias("shape_occurrence"))

    color_grouped = df.groupBy("year", "state", "color_reported") \
        .agg(count("*").alias("color_occurrence"))

    df_joined = shape_grouped.join(color_grouped,
                                on=["year", "state"],
                                how="inner")
    
    return df_joined


#Rank based on colour and shape occurances

def create_final_dataframe(df):
    """Create final dataframe"""
    shape_window_spec = Window.partitionBy("year", "state").orderBy(col("shape_occurrence").desc()) 
    color_window_spec = Window.partitionBy("year", "state").orderBy(col("color_occurrence").desc())

    # Selecting top occurrences of shape and color per year and state
    final_df = df.withColumn("shape_rank", row_number().over(shape_window_spec)) \
        .withColumn("color_rank", row_number().over(color_window_spec)) \
        .filter((col("shape_rank") == 1) & (col("color_rank") == 1)) \
        .select("year", "state", "shape_reported", "shape_occurrence", "color_reported", "color_occurrence","shape_rank","color_rank") \
        .orderBy(col("shape_occurrence").desc())
    
    return final_df

df_prepared = prepare_dataframe(df_spark)
df_joined = join_dataframes(df_prepared)
df_final = create_final_dataframe(df_joined)

timestamp_target = datetime.now().strftime("%Y%m%d_%H%M%S")
s3_path = f"s3://aws-glue-s3-bucket/ufo_reports_target_csv/{timestamp_target}/"

df_final.coalesce(1).write.mode("overwrite").csv(s3_path, header=True)

# From Spark dataframe to glue dynamic frame
glue_dynamic_frame_final = DynamicFrame.fromDF(df_final, glueContext, "glue_etl")

# Write the data in the DynamicFrame to a location in Amazon S3 and a table for it in the AWS Glue Data Catalog
s3output = glueContext.getSink(
  path="s3://aws-glue-s3-bucket/ufo_reports_target_paraquet/",
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

s3output.setFormat("glueparquet")
s3output.writeFrame(glue_dynamic_frame_final)

job.commit()

