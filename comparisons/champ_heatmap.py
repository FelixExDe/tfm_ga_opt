import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------
# 1. Cargar y Procesar Datos desde el Fichero CSV
# ------------------------------------------------------------------

# Cargar el dataset completo
try:
    df_full = pd.read_csv('champions_comparison_base_vs_optimized.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que el fichero 'champions_comparison_base_vs_optimized.csv' está en el mismo directorio.")
    exit()

# Definir las métricas del heatmap y su correspondencia con las columnas del CSV
metrics_heatmap = ["Acc", "MP", "MR", "P-F", "P-NR", "P-NF", "R-F", "R-NR", "R-NF"]
column_mapping = {
    'Accuracy': 'Acc',
    'Macro_Precision': 'MP',
    'Macro_Recall': 'MR',
    'Precision_functional': 'P-F',
    'Precision_functional_needs_repair': 'P-NR',
    'Precision_non_functional': 'P-NF',
    'Recall_functional': 'R-F',
    'Recall_functional_needs_repair': 'R-NR',
    'Recall_non_functional': 'R-NF'
}

# Crear una función para procesar los dataframes de forma consistente
def process_dataframe(df_input):
    """Prepara el dataframe para la comparación."""
    df = df_input.copy()
    # Estandarizar las columnas que formarán el índice
    df['Algorithm'] = df['Algorithm'].replace({'LOGISTIC': 'RL', 'RANDOM_FOREST': 'RF'})
    df['Data_Version'] = 'v' + df['Data_Version'].astype(str)
    df['Resampling'] = df['Resampling'].replace({'no_resample': 'No', 'resampled': 'Sí'})

    # --- INICIO DE CAMBIOS PARA REORDENAR ---
    # Convertir la columna 'Algorithm' a un tipo Categórico con un orden específico.
    # Esto asegura que .sort_index() respetará este orden en lugar del alfabético.
    algo_order = ['RL', 'KNN', 'RF']
    df['Algorithm'] = pd.Categorical(df['Algorithm'], categories=algo_order, ordered=True)
    # --- FIN DE CAMBIOS ---
    
    # Establecer el índice múltiple para alinear los datos
    df = df.set_index(['Algorithm', 'Data_Version', 'Resampling'])
    
    # Seleccionar solo las columnas necesarias y renombrarlas
    df_processed = df[column_mapping.keys()].rename(columns=column_mapping)
    
    # Asegurar que las columnas están en el orden deseado
    return df_processed[metrics_heatmap]

# Separar los datos en "baseline" y "optimized"
df_base_raw = df_full[df_full['Model_Type'] == 'baseline']
df_opt_raw = df_full[df_full['Model_Type'] == 'optimized']

# Procesar ambos dataframes
df_base = process_dataframe(df_base_raw)
df_opt = process_dataframe(df_opt_raw)

# Ordenar por el índice para asegurar una resta correcta
# Ahora .sort_index() usará el orden ['RL', 'KNN', 'RF'] que definimos.
df_base = df_base.sort_index()
df_opt = df_opt.sort_index()

# Calcular las diferencias (Optimizado - Base)
df_pct_change = ((df_opt - df_base) / df_base) * 100
df_pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
df_pct_change.fillna(0, inplace=True)

# Crear etiquetas legibles para las filas del heatmap
idx_labels = [f"{m}-{v}{'-ADA' if ada=='Sí' else ''}" for (m, v, ada) in df_pct_change.index]
df_pct_change.index = idx_labels

print("\nDiferencias (Optimizado − Base) en escala 0–1 (ej. +0.05 = +5 pp):")
print(df_pct_change.round(4))


# ------------------------------------------------------------------
# 2. Heatmap (Esta sección no necesita cambios)
# -------------------------------------------------------------------
data = df_pct_change.values

fig, ax = plt.subplots(figsize=(10, 7))

# Crear una paleta de color personalizada (Rojo -> Blanco -> Verde)
colors = ['#b2182b', 'white', '#1b7837']
cmap_rwg = LinearSegmentedColormap.from_list('red_white_green', colors)

# Centrar la paleta en 0
limit = np.abs(data).max()
im = ax.imshow(data, aspect='auto', cmap=cmap_rwg, vmin=-limit, vmax=limit)

# Ticks y etiquetas de los ejes
ax.set_xticks(np.arange(len(metrics_heatmap)))
ax.set_xticklabels(metrics_heatmap, rotation=45, ha='right', fontsize=12)
ax.set_yticks(np.arange(len(idx_labels)))
ax.set_yticklabels(idx_labels, fontsize=12)

# Anotar celdas con color de texto dinámico
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        text_color = 'white' if abs(val) > limit * 0.55 else 'black'
        ax.text(j, i, f"{val:+0.02f}", ha='center', va='center', fontsize=14, color=text_color)

# Colorbar y título
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Δ (Optimizado − Base)", rotation=-90, va='bottom', fontsize=12)
plt.tight_layout()
plt.show()