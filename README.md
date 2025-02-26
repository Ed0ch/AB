A/B Testing y Priorización de Hipótesis en una Tienda en Línea

📌 Descripción del Proyecto

Este proyecto analiza hipótesis de marketing para aumentar ingresos en una tienda en línea. Se priorizan hipótesis usando los frameworks ICE y RICE, y se realiza un test A/B para evaluar su impacto en ventas.

📂 Datos Utilizados

📊 Priorización de Hipótesis

hypotheses_us.csv: Lista de hipótesis con métricas de alcance, impacto, confianza y esfuerzo.

🛒 Test A/B

orders_us.csv: Datos de pedidos, ingresos y grupo asignado.

visits_us.csv: Registros de visitas diarias por grupo.

📌 Nota: Se procesaron datos para corregir errores y anomalías.

🚀 Metodología

Priorización de hipótesis con ICE y RICE.

Análisis de test A/B con visualización de ingresos, tasas de conversión y detección de anomalías.

Pruebas estadísticas para evaluar significancia y tomar decisiones.

📌 Tecnologías Utilizadas

Python (pandas, numpy, scipy, matplotlib, seaborn)

Estadística Inferencial (pruebas t, percentiles, conversión)

Frameworks de Priorización (ICE, RICE)

📈 Resultados y Conclusiones

📌 Resumen de hallazgos:

ICE y RICE priorizan diferente debido al alcance.

La tasa del grupo B mostró un aumento constante y significativo, superando a la del grupo A.

Se identificaron outliers en pedidos.

La diferencia en conversión fue estadísticamente significativa.

Decisión final: [Aceptar/Rechazar] la hipótesis y continuar/terminar el experimento.