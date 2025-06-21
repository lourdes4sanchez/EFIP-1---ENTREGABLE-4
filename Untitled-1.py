# ==========================================================
# Entregable 4 - EFIP 1 - Ciencia de Datos
# Visualización avanzada de datos - Análisis de vinos
# Autor: Lourdes Sanchez Alfaro
# Fecha: 29/06/2025
# ==========================================================

# ========== 1. IMPORTACIÓN DE LIBRERÍAS ==========
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

sns.set(style='whitegrid', palette='deep')

# ========== FUNCIÓN PARA GUARDAR Y MOSTRAR ==========
def guardar_y_mostrar(nombre_archivo):
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.show()

# ========== 2. CARGA Y DEPURACIÓN DE DATOS ==========
file_path = 'casoefip.csv'
df = pd.read_csv(file_path, header=None)
df = df[0].str.split(",", expand=True)
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['points', 'price'])

df = df[['country', 'description', 'points', 'price', 'province', 'variety', 'winery']]
df = df.dropna(subset=['country', 'description', 'points', 'price', 'variety'])
df['points'] = df['points'].astype(float)
df['price'] = df['price'].astype(float)

# ========== 3. EXPLORACIÓN INICIAL ==========
print(df.describe())
print(df['country'].value_counts().head(10))

# ========== 4. VISUALIZACIONES AVANZADAS ==========

# 4.1 - Top 10 países por puntuación promedio
plt.figure(figsize=(10,6))
sns.barplot(x=df.groupby('country')['points'].mean().sort_values(ascending=False).head(10).index,
            y=df.groupby('country')['points'].mean().sort_values(ascending=False).head(10).values,
            palette='viridis')
plt.title('Top 10 países por puntuación promedio')
plt.ylabel('Puntuación media')
plt.xlabel('País')
plt.xticks(rotation=45)
guardar_y_mostrar("grafico_top_paises.png")

# 4.2 - Relación entre precio y puntuación
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='price', y='points', alpha=0.3, color='teal')
plt.title('Relación entre Precio y Puntuación')
plt.xlabel('Precio (USD)')
plt.ylabel('Puntos')
plt.xscale('log')
guardar_y_mostrar("relacion_precio_puntuacion.png")

# 4.3 - Violin plot de puntuaciones por variedad (top 10)
top_varieties = df['variety'].value_counts().head(10).index
plt.figure(figsize=(12,6))
sns.violinplot(data=df[df['variety'].isin(top_varieties)], x='variety', y='points', palette='Set2')
plt.title('Distribución de puntuaciones por variedad (Top 10)')
plt.xticks(rotation=45)
guardar_y_mostrar("violin_variedad.png")

# 4.4 - Distribución de precios
plt.figure(figsize=(10,6))
sns.histplot(df['price'], bins=50, kde=True, color='salmon')
plt.xscale('log')
plt.title('Distribución de precios de vino')
plt.xlabel('Precio (USD)')
plt.ylabel('Frecuencia')
guardar_y_mostrar("distribucion_precios.png")

# 4.5 - Nube de palabras de descripciones
cantidad = min(10000, df['description'].dropna().shape[0])
text = ' '.join(df['description'].dropna().sample(cantidad))
wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='inferno').generate(text)
plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras de descripciones de vinos')
guardar_y_mostrar("nube_palabras.png")

# 4.6 - Relación precio vs. puntuación con densidad (hexbin)
sns.jointplot(data=df, x='price', y='points', kind='hex', height=8, color='purple')
plt.suptitle('Densidad de Precio vs. Puntos')
plt.tight_layout()
plt.savefig("hexbin_precio_puntos.png", dpi=300)
plt.show()

# 4.7 - Mejores variedades calidad/precio (con etiquetas de valores en las barras)
variety_stats = df.groupby('variety').agg({'price': 'mean', 'points': 'mean', 'variety': 'count'}).rename(columns={'variety': 'count'})
variety_stats = variety_stats[variety_stats['count'] > 100]
variety_stats['precio_por_punto'] = variety_stats['price'] / variety_stats['points']
best_value_varieties = variety_stats.sort_values('precio_por_punto').head(10)

plt.figure(figsize=(10,6))
barplot = sns.barplot(x=best_value_varieties.index, y=best_value_varieties['precio_por_punto'], palette='coolwarm')
plt.title('Top 10 variedades con mejor relación calidad/precio')
plt.ylabel('USD por punto')
plt.xlabel('Variedad')
plt.xticks(rotation=45)
for i, val in enumerate(best_value_varieties['precio_por_punto']):
    barplot.text(i, val + 0.2, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
guardar_y_mostrar("variedad_valor.png")

# 4.8 - Países con más vinos premium (≥95 puntos)
premium_wines = df[df['points'] >= 95]
top_premium = premium_wines['country'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_premium.index, y=top_premium.values, palette='magma')
plt.title('Países con más vinos premium (≥95 puntos)')
plt.ylabel('Cantidad de vinos')
plt.xlabel('País')
plt.xticks(rotation=45)
guardar_y_mostrar("vinos_premium.png")

# 4.9 - Gráfico de torta (pie chart) de reseñas por país (Top 5)
top5_paises = df['country'].value_counts().head(5)
plt.figure(figsize=(7, 7))
plt.pie(top5_paises, labels=top5_paises.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Distribución de reseñas por país (Top 5)")
plt.axis('equal')
guardar_y_mostrar("torta_paises.png")

# 4.10 - Boxplot de puntuaciones por país (Top 5)
top5_country = df['country'].value_counts().head(5).index
plt.figure(figsize=(10,6))
sns.boxplot(data=df[df['country'].isin(top5_country)], x='country', y='points', palette='Accent')
plt.title('Distribución de puntuaciones por país (Top 5)')
plt.ylabel('Puntos')
plt.xlabel('País')
plt.xticks(rotation=45)
guardar_y_mostrar("boxplot_paises.png")

# Mapa geográfico de puntuación promedio por país
country_avg = df.groupby('country')['points'].mean().reset_index()
fig = px.choropleth(country_avg,
                    locations='country',
                    locationmode='country names',
                    color='points',
                    title='Puntuación promedio por país',
                    color_continuous_scale='Viridis')
fig.write_image("mapa_puntuacion.png")
fig.show()

# ========== 5. MÉTRICAS FINALES ==========
print("Cantidad total de vinos analizados:", len(df))
print("Promedio general de puntuación:", round(df['points'].mean(), 2))
print("Promedio general de precio (USD):", round(df['price'].mean(), 2))
print("\\n--- Análisis finalizado con éxito ---")
