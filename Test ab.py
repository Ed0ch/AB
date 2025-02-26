#  Test A/B
 
# En este proyecto estaremos llevando a cabo en analisis de una gran tienda, en conjunto con el departamento de marketing hemos recopilado una lista de hipótesis que nos pueden ayudar a aumentar los ingresos, por lo que estaremos realizando distintas pruebas para priorizar las hipótesis, lanzar un test A/B y analizar los resultados.
# 
# 

# ## Importación de librerías




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as stats
import numpy as np


# ## Creación de dataframes




df_hypotheses=pd.read_csv("C:/Users/carlo/OneDrive/Documentos/Marca personal/AB/hypotheses_us.csv",sep=';') #Originalmente el dataset tenía las columnas separadas por ; 
df_orders=pd.read_csv("C:/Users/carlo/OneDrive/Documentos/Marca personal/AB/orders_us.csv")
df_visits=pd.read_csv("C:/Users/carlo/OneDrive/Documentos/Marca personal/AB/visits_us.csv")




# ## Revisión y preprocesamiento de datos

# ### Dataframe de hipotésis




print(df_hypotheses.head()) #Revisión de las columnas, podemos observar que están en mayúsculas los títulos





df_hypotheses.columns=df_hypotheses.columns.str.lower()
print(df_hypotheses.head())





print(df_hypotheses.info()) #Revisión de formatos, valores ausentes


# Al ser un dataframe de hipótesis podemos observar que el contenido de los datos en su mayoría es numerico, se tenían ciertos errores como lo fue que estuviesen separadas las columnas por un punto y coma, así como que los titulos tuviesen mayusculas en su mayoría.
# No se tienen duplicados en este caso o valores ausentes por lo que podemos continuar con este dataframe para trabajar.
# 

# ### Dataframe de Ordenes




print(df_orders.head()) #Revisión del título de las columnas, podemos observar que las primeras dos tienen errores





df_orders.rename(columns={'transactionId':'transaction_id','visitorId':'visitor_id'},inplace=True) #Corrección de nombre de las columnas
print(df_orders.head())





print(df_orders.info()) #Revisión de formatos de los datos





df_orders.drop_duplicates(subset=['visitor_id','group'],inplace=True) #Eliminación de duplicados 





df_orders['date']=pd.to_datetime(df_orders['date']) #Convertir la columna a formato de fecha
print(df_orders.info())


# Revisando el dataframe de las ordenes podemos ver que teníamos el nombre de las columnas de transacción y de visitantes de una manera erronea, por lo que decidí realizarles una separación para poder hacerlo más entendible, así como pusimos la columna de fecha en el formato correspondiente, por otro lado revisamos de que nuestro dataframe no tuviese ningun duplicado o valor ausente.
# 




print(df_visits.head())





print(df_visits.info()) #Revisión de formatos de los datos





df_visits['date']=pd.to_datetime(df_visits['date']) #Conversión a formato de fecha.
print(df_visits.info())




# ## Priorizar hipótesis
# 
# En esta parte estaremos utilizando el dataframe de hipótesis, el cual contiene nueve hipótesis sobre cómo aumentaar los ingresos de una tienda en línea.

# ### ICE SCORE




print(df_hypotheses.head())





df_hypotheses['ICE']=(df_hypotheses['impact']*df_hypotheses['confidence'])/df_hypotheses['effort']

print(df_hypotheses[['hypothesis','ICE']].sort_values(by='ICE',ascending=False))


# Podemos ver que la hipotésis con mayor puntuage en el ICE scpre viene siendo la numero 8 con 16 puntos, le sigue la 0 con 13 puntos y luego la 7 con once.
# Ahora estaremos comparando las hipótesis con el método RICE en la parte de abajo.
# 

# ### RICE SCORE




df_hypotheses['RICE']=(df_hypotheses['reach']*df_hypotheses['impact']*df_hypotheses['confidence'])/df_hypotheses['effort']

print(df_hypotheses[['hypothesis','RICE']].sort_values(by='RICE',ascending=False))


# Aqui podemos revisar que el orden de las hipotesis con el método RICE es el siguiente:
# 
# 1.- Hipótesis 7 con 112 puntos.
# 
# 2.-Hipótesis 2 con 56 puntos.
# 
# 3.- Hipótesis 0 con 40 puntos.
# 

# ### Comparación de resultados entre ambos métodos

# En este caso podemos obsverar que el componente que le agregar el método RICE, que viene siendo el alcance, hace una gran modificación a la hora de priorizar las hipótesis en nuestra comparativa, si observamos la hipótesis 8 tenía el mayor puntuaje, por lo que se fue a la primera posición en el método ICE pero al compararlo con el otro se fue hasta las cuarta posición y la hipótesis 7 fue las que se siguió manteniendo en el top 3 en ambos, en este caso por método RICE quedó en la primera posición, dándonos a entender que debemos priorizarla, así como la hipótesis 0 que también se mantiene dentro de los primeros lugares.
# 



# ## Análisis del Test A/B

# ### Ingreso acumulado por grupo



date_groups=df_orders[['date','group']].drop_duplicates() #Aquie lo que queremos es crear una matriz con valores únicos de parejas de fehca y grupo





# Este código nos sirve para obtener los datos diarios acumulados agregados sobre los pedidos
orders_aggregated=date_groups.apply(lambda x:df_orders[np.logical_and(df_orders['date']<= x['date'],df_orders['group']==x['group'])].agg({'date':'max','group':'max','transaction_id':pd.Series.nunique,'visitor_id':pd.Series.nunique,'revenue':'sum'}),axis=1).sort_values(by=['date','group'])

# Este código nos sirve para obtener los datos acumulados agregados de las visitas
visitors_aggregated=date_groups.apply(lambda x:df_visits[np.logical_and(df_visits['date']<= x['date'],df_visits['group']==x['group'])].agg({'date':'max','group':'max','visits':'sum'}),axis=1).sort_values(by=['date','group'])





#Hay que crear una variable para poder fusionarlos

cumulative_data=orders_aggregated.merge(visitors_aggregated,left_on=['date','group'],right_on=['date','group'])
cumulative_data.columns=['date','group','orders','buyers','revenue','visits']

print(cumulative_data.head())


# ### Gráficos



plt.figure(figsize=(10, 6))

cumulative_revenueA=cumulative_data[cumulative_data['group']=='A'][['date','revenue','orders']]

cumulative_revenueB=cumulative_data[cumulative_data['group']=='B'][['date','revenue','orders']]

plt.plot(cumulative_revenueA['date'],cumulative_revenueA['revenue'],label='A')

plt.plot(cumulative_revenueB['date'],cumulative_revenueB['revenue'],label='B')



plt.legend()


# En el gráfico podemos observar que el grupo A y el B tienen un inicio muy similar, en donde el A siempre va a la cabeza sobre el grupo B y después del 2019-08-21 podemos ver que el grupo A tiene un repunte muy sobresaliente sobre el grupo B y ahí es donde comienza a subir exponencialmente a diferencia del B, en el cual sigue subiendo de manera lineal.
# En este caso podemos ver que los ingresos siguen aumentando constantemente durante toda la prueba, pese a eso podemos ver que el grupo A es más evidente que tiene picos en sus pedidos, lo cual nos indica que hay un gran aumento en la cantidad de pedidos o que hay pedidos muy costosos.
# 

# ### Tamaño de pedido acumulado
# 




plt.figure(figsize=(10,6))
plt.plot(cumulative_revenueA['date'],cumulative_revenueA['revenue']/cumulative_revenueA['orders'],label='A')
plt.plot(cumulative_revenueB['date'],cumulative_revenueB['revenue']/cumulative_revenueB['orders'],label='B')
plt.legend()


# Podemos ver que el tamaño promedio de compra parece indicar que se comienza a estabilizar ligaremente al último en el grupo A, pero en el caso del grupo B parece ser que comienza a decaer con una ligera estabilización pero no aún para decir que ya tiene un nivel.

# ### Diferencia relativa en el tamaño promedio de pedido




#Reunir los datos en un Dataframe
merged_revenue=cumulative_revenueA.merge(cumulative_revenueB,left_on='date',right_on='date',how='left',suffixes=['A','B'])

#Trazar un gráfico de diferencia relativa para los tamaños de compra promedio
plt.figure(figsize=(10,6))
plt.plot(merged_revenue['date'],(merged_revenue['revenueB']/merged_revenue['ordersB'])/(merged_revenue['revenueA']/merged_revenue['ordersA'])-1)

#agregar el eje X
plt.axhline(y=0,color='black',linestyle='--')


# ### Conversión acumulada de cada grupo




cumulative_data['conversion']=cumulative_data['orders']/cumulative_data['visits'] #Calcular la conversión acumulada

cumulative_dataA=cumulative_data[cumulative_data['group']=='A']

cumulative_dataB=cumulative_data[cumulative_data['group']=='B']
plt.figure(figsize=(10,6))
plt.plot(cumulative_dataA['date'],cumulative_dataA['conversion'],label='A')
plt.plot(cumulative_dataB['date'],cumulative_dataB['conversion'],label='B')
plt.legend()



# En este caso podemos ver que los gráficos pese a que son bastante similares en comportamiento, no son simétricos, lo cual nos indica que nuestros datos son fiables, podmeos ver que al inicio los datos fluctuaron alrededor del mismo valor pero la tasa de conversión del B comenzó su trayectoria hacia el alza con cambios o picos significativos, mientras que el grupo A después de su fuerte inicio comenzó su caída para luego comenzar a subir lentamente. 
# La tasa de conversión del B es mucho más fuerte que la tasa de conversión del grupo A.

# ### Número de pedidos por usuario
# 




orders_users=(df_orders.drop(['group','revenue','date'],axis=1).groupby('visitor_id',as_index=False).agg({'transaction_id':pd.Series.nunique}))
orders_users.columns=['visitor_id','transaction_id']

orders_users.sort_values(by='transaction_id',ascending=False)





pedidos=pd.Series(range(0,len(orders_users)))

plt.scatter(pedidos,orders_users['transaction_id'])


# En este caso podemos ver que hay muchos usuarios con dos pedidos, aun no sabemos que proporción exactamente es la que tienen esos dos o si debemos considerar los otros como anomalías por lo que abajo estaremos calculando los percentiles para el número de pedidos por usuario.

# ### Percentiles para el número de pedidos por usuario




print(np.percentile(orders_users['transaction_id'],[90,95,99]))


# En esta caso podemos ver que el numero de pedidos por usuario en ambos casos es de dos, por lo que podemos decir que no más del 5% de los pedidos de usuario es de dos

# ### Gráfico de dispersión para el precio  de  los pedidos




scatter_values=pd.Series(range(0,len(df_orders['revenue'])))
plt.figure(figsize=(10,6))
plt.scatter(scatter_values,df_orders['revenue'])


# Aquí podemos observar que no tenemos tantos valores atípicos, se puede ver que hay un valor allá por $20000 y otro casi por los $5000, pero no representan la gran mayoría de los datos, si tenemos algunos valores que se despegan un poco más que los otros, los cuales son los que estaremos revisando en el análisis exploratorio de los datos.

# ### Percentiles para el precio de  los pedidos por usuario




print(np.percentile(df_orders['revenue'],[95,99]))


# Estos valores nos dan las siguiente conjeturas acerca de nuestros pedidos: No más del 5% cuestan más de  431, no mas del 1 porciento  cuestan mas de 908.00, por lo que podemos deducir que los puntos que se ven en nuestro diagrama cercano al 5000 y $ 20000 son anomalías.

# ### Análisis de los datos en bruto-Significancia estadística

# ####  Diferencia de conversión entre los grupos




#Vamos a crear dos variables para almacenar las columnas y poder indicar los pedidos
ordersbyusersA=df_orders[df_orders['group']=='A'].groupby('visitor_id',as_index=False).agg({'transaction_id':pd.Series.nunique})
ordersbyusersA.columns=['visitor_id','transaction_id']

ordersbyusersB=df_orders[df_orders['group']=='B'].groupby('visitor_id',as_index=False).agg({'transaction_id':pd.Series.nunique})
ordersbyusersB.columns=['visitor_id','transaction_id']

#Tenemos que crear las muestras con usuarios de diferentes grupos y pedidos correspondientes para poder pasar a la prueba de Mann-Whitney.

sampleA=pd.concat([ordersbyusersA['transaction_id'],pd.Series(0,index=np.arange(df_visits[df_visits['group']=='A']['visits'].sum()-len(ordersbyusersA['transaction_id'])),name='transaction_id')],axis=0)
sampleB=pd.concat([ordersbyusersB['transaction_id'],pd.Series(0,index=np.arange(df_visits[df_visits['group']=='B']['visits'].sum()-len(ordersbyusersB['transaction_id'])),name='transaction_id')],axis=0)

#Pasamos a calcular la prueba de Mann-Whitney

print('{0:.3f}'.format(stats.mannwhitneyu(sampleA,sampleB)[1]))
print('{0:.3f}'.format(sampleB.mean()/sampleA.mean()-1)) 


# Dado que el valor de p es menor que el nivel de significancia que viene siendo de 0.05, tendríamos evidencia suficiente para rechazar la hipótesis nula. En otras palabras, podríamos concluir que hay diferencias significativas entre las dos muestras que estamos comparando en las muestras de los grupos A y B.
# Por otro lado hay una ganancia relativa del grupo B del 15.4%.

# #### Diferencia en el tamaño promedio de pedido




print('{0:.3f}'.format(stats.mannwhitneyu(df_orders[df_orders['group']=='A']['revenue'],df_orders[df_orders['group']=='B']['revenue'])[1]))
print('{0:.3f}'.format(df_orders[df_orders['group']=='B']['revenue'].mean()/df_orders[df_orders['group']=='A']['revenue'].mean()-1))


# El valor de p es mayor notablemente a 0.05, por lo que no hay motivo para rechazar la hipótesis nula y concluir que el tamaño de pedidos difiere entre los grupos, por otro lado el tamaño de pedido promedio para el grupo B es mucho más grande que para el grupo A.

# ### Análisis de los datos filtrados-Significancia estadística

# #### Usuarios anómalos




users_manyorders=pd.concat([ordersbyusersA[ordersbyusersA['transaction_id']>1]['visitor_id'],ordersbyusersB[ordersbyusersB['transaction_id']>1]['visitor_id']],axis=0)
users_expensiveorders=df_orders[df_orders['revenue']>500]['visitor_id']

abnormal_users=pd.concat([users_manyorders,users_expensiveorders],axis=0).drop_duplicates().sort_values()

print(abnormal_users.head())
print(abnormal_users.shape)


# En total tenemos 41 usuarios anomalos, en la parte de abajo estaremos averiguando como sus acciones afectan los resultados de la prueba.

# #### Significancia estadística de las diferencias en la conversión entre los grupos con datos filtrados




sampleAfiltered=pd.concat([ordersbyusersA[np.logical_not(ordersbyusersA['visitor_id'].isin(abnormal_users))]['transaction_id'],pd.Series(0,index=np.arange(df_visits[df_visits['group']=='A']['visits'].sum()-len(ordersbyusersA['transaction_id'])),name='transaction_id')],axis=0)
sampleBfiltered=pd.concat([ordersbyusersB[np.logical_not(ordersbyusersB['visitor_id'].isin(abnormal_users))]['transaction_id'],pd.Series(0,index=np.arange(df_visits[df_visits['group']=='B']['visits'].sum()-len(ordersbyusersB['transaction_id'])),name='transaction_id')],axis=0)

print("{0:.3f}".format(stats.mannwhitneyu(sampleAfiltered, sampleBfiltered)[1]))
print("{0:.3f}".format(sampleBfiltered.mean()/sampleAfiltered.mean()-1)) 


# Los resultados de conversión casi no cambiaron entre varios grupos

# #### Significancia estadística de las diferencias entre el tamaño promedio de los pedidos con los  datos filtrados




print('{0:.3f}'.format(stats.mannwhitneyu(
    df_orders[np.logical_and(
        df_orders['group']=='A',
        np.logical_not(df_orders['visitor_id'].isin(abnormal_users)))]['revenue'],
    df_orders[np.logical_and(
        df_orders['group']=='B',
        np.logical_not(df_orders['visitor_id'].isin(abnormal_users)))]['revenue'])[1]))

print('{0:.3f}'.format(
    df_orders[np.logical_and(df_orders['group']=='B',np.logical_not(df_orders['visitor_id'].isin(abnormal_users)))]['revenue'].mean()/
    df_orders[np.logical_and(
        df_orders['group']=='A',
        np.logical_not(df_orders['visitor_id'].isin(abnormal_users)))]['revenue'].mean() - 1)) 


# Podemos ver que el valor de P aumenta un poco pero que la diferencia entre los segmentos es del 1.1% en vez del 26%, esto nos permite ver que las anomalías si afectan a los resultados.

# ## Conclusión

# Basándonos en estos hechos realizados en el análisis podemos concluir que la prueba fue exitosa y debe continuar, ya que la probabilidad de que el segmento B ha resultado mejor que el segmento A es significativamente notorio.


