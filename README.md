#Executando o projeto


#pip install -r requirements.txt


#definir variaveis os.environ["SPARK_HOME"], os.environ["JAVA_HOME"]

```Python

 als = ALS(userCol="user_id", itemCol="item_id", ratingCol="price", coldStartStrategy="drop")
    model = als.fit(data)

    userRecs = model.recommendForAllUsers(10)

    userRecs = userRecs.withColumnRenamed("recommendations", "recommendation")
    userRecs = userRecs.select("user_id", "recommendation.item_id", "recommendation.rating")

    data = data.withColumn("shipping_date", date_format(col("shipping_limit_date"), "yyyy-MM-dd"))
    data = data.select("order_id", "user_id", "customer_id", "item_id", "price", "shipping_date", "product_category_name")

    userRecs = userRecs.join(data, on="user_id").drop("user_id")

```

<b>Modelo ALS 

Alternating Least Square é um algoritmo de fatoração de matriz implementado no Apache Spark ML e construído para problemas de filtragem colaborativa em larga escala.

Fatoração de matrizes (ou decomposição)
A ideia básica é decompor uma matriz em partes menores da mesma forma que podemos fazer para um número. Por exemplo, podemos dizer que o número quatro pode ser decomposto em dois vezes dois (4 = 2 x 2). Da mesma forma, podemos fazer uma decomposição de uma matriz.

Mais detalhes: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html
</n>
