from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, date_format
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler


"""
    Import o arquivo CSV, ou se precisar, altere os campos para os campos do csv que for usar, a lógica praticamente nao muda
    Esta sendo usado als hibrido
"""

def create_spark_session(app_name: str) -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark

def load_data(spark: SparkSession, csv_file_path: str, delimiter: str) -> 'DataFrame':
    data = spark.read.option("header", True).option("delimiter", delimiter).csv(csv_file_path)
    data = data.toDF(*(col_name.replace(' ', '_') for col_name in data.columns))
    
    numeric_cols = ['price', 'freight_value', 'payment_value', 'product_weight_g'
                    , 'product_height_cm', 'product_width_cm', 'payment_sequential']
    for col_name in numeric_cols:
        data = data.withColumn(col_name, regexp_replace(col(col_name), ',', '.').cast('float'))
    
    data = data.withColumn('product_name_lenght', col('product_name_lenght').cast('int'))
    
    return data

def create_index_mapping(data, input_col):
    distinct_values = data.select(input_col).distinct().rdd.map(lambda x: x[0]).collect()
    index_mapping = {value: index for index, value in enumerate(distinct_values)}
    return index_mapping

def main():
    spark = create_spark_session("RecomendaçãoALS")

    csv_file_path = r"D:\teste2607\pedidos_olist.csv"
    data = load_data(spark, csv_file_path, ';')

    indexers = [
        StringIndexer(inputCol="order_id", outputCol="user_id", handleInvalid="keep"),
        StringIndexer(inputCol="product_id", outputCol="item_id", handleInvalid="keep")
    ]

    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(data)
    data = model.transform(data)

    als = ALS(userCol="user_id", itemCol="item_id", ratingCol="price", coldStartStrategy="drop")
    model = als.fit(data)

    userRecs = model.recommendForAllUsers(10)

    userRecs = userRecs.withColumnRenamed("recommendations", "recommendation")
    userRecs = userRecs.select("user_id", "recommendation.item_id", "recommendation.rating")

    data = data.withColumn("shipping_date", date_format(col("shipping_limit_date"), "yyyy-MM-dd"))
    data = data.select("order_id", "user_id", "customer_id", "item_id", "price"
                       , "shipping_date", "product_category_name")

    userRecs = userRecs.join(data, on="user_id").drop("user_id")

    userRecs.show(truncate=False)



"""Implementando"""
def run_tasks():
    # Crie o objeto scheduler
    scheduler = BlockingScheduler()

    # Agende a tarefa para executar o ALS a cada hora
    scheduler.add_job(run_als_algorithm, trigger='interval', hours=1)


    # Inicie o scheduler
    try:
        print("Scheduler iniciado. Pressione Ctrl+C para finalizar.")
        scheduler.start()
    except KeyboardInterrupt:
        print("Scheduler interrompido pelo usuário.")



if __name__ == "__main__":
    main()
