#!/usr/bin/env python3
"""
Estudio Estadístico Inicial - TFM Detección Lesiones Esclerosis Múltiple
=======================================================================
Fase 01: Análisis descriptivo de datasets ImaginEM (349 IDs) y MSSEG2 (40 IDs training)

Se calcula:
- Número de lesiones y volúmenes (mm3)
- Comparación volúmenes calculados vs CSV (ImaginEM)
- Verificación etiquetas únicas por dataset
- Tablas resumen, histogramas, boxplots y balanceo de clases

Resultados guardados en: outputs/01_estadisticas/
"""
# Se importan los paquetes necesarios
import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label
import matplotlib.pyplot as plt
from pathlib import Path

# Se configuran las rutas
BASE_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs/01_estadisticas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGINEM_DIR = BASE_DIR / "NEW_LESIONS_IMAGINEM"
MSSEG2_DIR = BASE_DIR / "MSSEG-2" / "LongitudinalMultipleSclerosisLesionSegmentationChallengeMiccai21_v2" / "training"
IMAGINEM_CSV = BASE_DIR / "ImaginEM_nLS.csv"

# Para el estilo de las representaciones
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150

# Función que verifica que las etiquetas de cada dataset estén en orden
def verificar_etiquetas(dataset_dir, nombre_dataset, labels_esperados):
    """Verifica etiquetas únicas en segmentaciones manuales."""
    print(f"\n===== VERIFICACIÓN ETIQUETAS - {nombre_dataset} =====")

    labels_por_paciente = {}
    for folder in tqdm(sorted(os.listdir(dataset_dir)), desc="Verificando etiquetas"):
        folder_path = dataset_dir / folder
        if not folder_path.is_dir():
            continue

        seg_path = None
        if nombre_dataset == "IMAGINEM":
            seg_path = folder_path / f"{folder}_seg.nii.gz"
        elif nombre_dataset == "MSSEG2":
            seg_path = folder_path / "ground_truth.nii.gz"

        if seg_path and seg_path.exists():
            data = nib.load(seg_path).get_fdata()
            labels_por_paciente[folder] = sorted(np.unique(data))

    # Estadísticas
    todas_etiquetas = np.concatenate(list(labels_por_paciente.values()))
    etiquetas_globales = np.unique(todas_etiquetas)

    print(f"Total pacientes analizados: {len(labels_por_paciente)}")
    print(f"Etiquetas únicas globales: {etiquetas_globales}")

    pacientes_invalidos = [p for p, labs in labels_por_paciente.items()
                          if not set(labs).issubset(labels_esperados)]
    print(f"Pacientes con labels fuera de {labels_esperados}: {len(pacientes_invalidos)}")

    return labels_por_paciente

# Estadísticas para ImaginEM
def compute_lesion_stats_imaginem(seg_path):
    """Calcula estadísticas lesiones ImaginEM (labels 1=estables, 2=nuevas)."""
    img = nib.load(seg_path)
    data = img.get_fdata()

    # Vóxeles por label
    vox_estables = np.sum(data == 1)
    vox_nuevas = np.sum(data == 2)

    # Volúmenes (mm3)
    voxel_volume = np.prod(img.header.get_zooms())
    vol_estables = vox_estables * voxel_volume
    vol_nuevas = vox_nuevas * voxel_volume

    # Número de lesiones
    n_estables, _ = label(data == 1)
    n_nuevas, _ = label(data == 2)
    n_les_estables = int(np.max(n_estables))
    n_les_nuevas = int(np.max(n_nuevas))

    return {
        "n_les_estables": n_les_estables,
        "n_les_nuevas": n_les_nuevas,
        "vol_les_estables_mm3": vol_estables,
        "vol_les_nuevas_mm3": vol_nuevas,
    }

# Estadísticas para MSSEG2
def compute_lesion_stats_msseg2(seg_path):
    """Calcula estadísticas MSSEG2 (solo label 1=nuevas)."""
    img = nib.load(seg_path)
    data = img.get_fdata()

    vox_nuevas = np.sum(data == 1)
    voxel_volume = np.prod(img.header.get_zooms())
    vol_nuevas = vox_nuevas * voxel_volume

    n_nuevas, _ = label(data == 1)
    n_les_nuevas = int(np.max(n_nuevas))

    return {
        "n_les_nuevas": n_les_nuevas,
        "vol_les_nuevas_mm3": vol_nuevas
    }

# Analisis estadístico de ImaginEM
def analizar_imaginem():
    """Análisis completo ImaginEM."""
    print("="*70)
    print("ANÁLISIS ESTADÍSTICO - IMAGINEM (349 IDs)")
    print("="*70)

    # Verificamos etiquetas
    labels_imag = verificar_etiquetas(IMAGINEM_DIR, "IMAGINEM", {0,1,2})

    # Cargamos CSV volúmenes
    df_vol_nuevas = pd.read_csv(IMAGINEM_CSV)

    # Calculamos estadísticas
    resultados = []
    for folder in tqdm(labels_imag.keys(), desc="Procesando ImaginEM"):
        seg_path = IMAGINEM_DIR / folder / f"{folder}_seg.nii.gz"
        if not seg_path.exists():
            continue

        stats = compute_lesion_stats_imaginem(seg_path)

        # Volumen CSV
        fila_csv = df_vol_nuevas[df_vol_nuevas["ID"] == f"{folder}_seg"]
        vol_csv = float(fila_csv.iloc[0]["volume_mm3"]) if len(fila_csv) > 0 else np.nan

        diff_vol = abs(stats["vol_les_nuevas_mm3"] - vol_csv) if not np.isnan(vol_csv) else np.nan

        resultados.append({
            "ID": folder,
            "n_les_estables": stats["n_les_estables"],
            "n_les_nuevas": stats["n_les_nuevas"],
            "vol_les_estables_mm3": stats["vol_les_estables_mm3"],
            "vol_les_nuevas_calc_mm3": stats["vol_les_nuevas_mm3"],
            "vol_les_nuevas_csv_mm3": vol_csv,
            "diferencia_volumen_mm3": diff_vol,
            "tiene_lesiones_nuevas": stats["n_les_nuevas"] > 0
        })

    df_imag = pd.DataFrame(resultados)
    df_imag.to_csv(OUTPUT_DIR / "estadisticas_imaginem_349IDs.csv", index=False)

    # Resumen estadístico
    total = len(df_imag)
    con_nuevas = df_imag["tiene_lesiones_nuevas"].sum()
    sin_nuevas = total - con_nuevas

    print(f"\nRESUMEN GENERAL:")
    print(f"Total pacientes: {total}")
    print(f"Con lesiones nuevas: {con_nuevas} ({con_nuevas/total*100:.1f}%)")
    print(f"Sin lesiones nuevas: {sin_nuevas} ({sin_nuevas/total*100:.1f}%)")

    print("\nLesiones nuevas - describe():")
    print(df_imag["n_les_nuevas"].describe())
    print(df_imag["vol_les_nuevas_calc_mm3"].describe())

    # Diferencias volumen
    print(f"\nDiferencias volumen >10mm³: {(df_imag['diferencia_volumen_mm3']>10).sum()}")
    print(f"Diferencias volumen >100mm³: {(df_imag['diferencia_volumen_mm3']>100).sum()}")

    return df_imag

# Análisis estadístico de MSSEG2
def analizar_msseg2():
    """Análisis completo MSSEG2."""
    print("\n"*2 + "="*70)
    print("ANÁLISIS ESTADÍSTICO - MSSEG2 (40 IDs training)")
    print("="*70)

    # Verificar etiquetas
    labels_msseg2 = verificar_etiquetas(MSSEG2_DIR, "MSSEG2", {0,1})

    # Calcular estadísticas
    resultados = []
    for folder in tqdm(labels_msseg2.keys(), desc="Procesando MSSEG2"):
        seg_path = MSSEG2_DIR / folder / "ground_truth.nii.gz"
        if not seg_path.exists():
            continue

        stats = compute_lesion_stats_msseg2(seg_path)
        resultados.append({
            "ID": folder,
            "n_les_nuevas": stats["n_les_nuevas"],
            "vol_les_nuevas_mm3": stats["vol_les_nuevas_mm3"],
            "tiene_lesiones_nuevas": stats["n_les_nuevas"] > 0
        })

    df_msseg2 = pd.DataFrame(resultados)
    df_msseg2.to_csv(OUTPUT_DIR / "estadisticas_msseg2_40IDs.csv", index=False)

    # Resumen
    total = len(df_msseg2)
    con_nuevas = df_msseg2["tiene_lesiones_nuevas"].sum()
    print(f"\nRESUMEN GENERAL:")
    print(f"Total pacientes: {total}")
    print(f"Con lesiones nuevas: {con_nuevas} ({con_nuevas/total*100:.1f}%)")

    print("\nLesiones nuevas - describe():")
    print(df_msseg2["n_les_nuevas"].describe())
    print(df_msseg2["vol_les_nuevas_mm3"].describe())

    return df_msseg2

# Se generan los histogramas y más representaciones
def generar_visualizaciones(df_imag, df_msseg2):
    """Genera y guarda todas las visualizaciones."""
    def mean_sd(series):
        return f"{series.mean():.2f} ± {series.std():.2f}"

    # Tabla resumen ImaginEM
    table_imag = pd.DataFrame({
        "Metric": ["Número lesiones [n]", "Volumen total [mm³]", "Casos sin lesiones [n]"],
        "Lesiones estables (1)": [
            mean_sd(df_imag["n_les_estables"]),
            mean_sd(df_imag["vol_les_estables_mm3"]),
            int((df_imag["n_les_estables"] == 0).sum())
        ],
        "Lesiones nuevas (2)": [
            mean_sd(df_imag["n_les_nuevas"]),
            mean_sd(df_imag["vol_les_nuevas_calc_mm3"]),
            int((df_imag["n_les_nuevas"] == 0).sum())
        ]
    })
    print("\nTABLA RESUMEN IMAGINEM:")
    print(table_imag.to_string(index=False))

    # Gráficos ImaginEM
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(df_imag["vol_les_nuevas_calc_mm3"], bins=30, alpha=0.7, color='lightsalmon')
    axes[0].set_title("Volumen lesiones nuevas (ImaginEM)")
    axes[0].set_xlabel("Volumen (mm³)")
    axes[0].set_ylabel("Nº pacientes")

    axes[1].hist(df_imag["n_les_nuevas"], bins=20, alpha=0.7, color='lightgreen')
    axes[1].set_title("Número lesiones nuevas (ImaginEM)")
    axes[1].set_xlabel("Nº lesiones")
    axes[1].set_ylabel("Nº pacientes")

    total_imag = len(df_imag)
    axes[2].bar(['Sin nuevas', 'Con nuevas'],
                [total_imag - df_imag["tiene_lesiones_nuevas"].sum(),
                 df_imag["tiene_lesiones_nuevas"].sum()],
                color=['lightcoral', 'lightblue'], alpha=0.8)
    axes[2].set_title("Balanceo ImaginEM")
    axes[2].set_ylabel("Nº pacientes")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "visualizaciones_imaginem.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gráficos MSSEG2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(df_msseg2["vol_les_nuevas_mm3"], bins=20, alpha=0.7, color='lightsalmon')
    axes[0].set_title("Volumen lesiones nuevas (MSSEG2)")

    axes[1].hist(df_msseg2["n_les_nuevas"], bins=15, alpha=0.7, color='lightgreen')
    axes[1].set_title("Número lesiones nuevas (MSSEG2)")

    total_msseg2 = len(df_msseg2)
    axes[2].bar(['Sin nuevas', 'Con nuevas'],
                [total_msseg2 - df_msseg2["tiene_lesiones_nuevas"].sum(),
                 df_msseg2["tiene_lesiones_nuevas"].sum()],
                color=['lightcoral', 'lightblue'])
    axes[2].set_title("Balanceo MSSEG2")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "visualizaciones_msseg2.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nGráficos guardados en {OUTPUT_DIR}")

if __name__ == "__main__":
    print("Iniciando estudio estadístico inicial...")

    # Se guarda en un df la info recogida
    df_imaginem = analizar_imaginem()
    df_msseg2 = analizar_msseg2()

    # Se generan las visualizaciones
    generar_visualizaciones(df_imaginem, df_msseg2)

    print(f"\n ANÁLISIS COMPLETADO!")
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print("- estadisticas_imaginem_349IDs.csv")
    print("- estadisticas_msseg2_40IDs.csv")
    print("- visualizaciones_imaginem.png")
    print("- visualizaciones_msseg2.png")
