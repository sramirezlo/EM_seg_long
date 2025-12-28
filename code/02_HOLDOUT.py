#!/usr/bin/env python3
"""
Holdout Split y Preprocesado nnU-Net v2 - TFM Detección Lesiones Esclerosis Múltiple
==================================================================================
Fase 02: División Holdout (70% train/15% validation/15 % test) y preprocesado Dataset202_ImaginEM_FLAIR_HOLDOUT

Proceso:
1. Crea Dataset102_ImaginEM_FLAIR (349 casos completos)
2. Divide en train(70%)/val(15%)/test(15%) con semilla 42
3. Crea Dataset202_ImaginEM_FLAIR_HOLDOUT (train+val = 85%)
4. Ejecuta nnUNetv2_plan_and_preprocess -d 202

Resultados: split_holdout.json, Dataset202 listo para entrenar
"""

import os
import json
import shutil
import random
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

plt.style.use('default')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_DIR = Path("data/raw/NEW_LESIONS_IMAGINEM")
NNUNET_RAW = Path("data/nnunet_raw")
NNUNET_PREPROCESSED = Path("data/nnunet_preprocessed")
NNUNET_RESULTS = Path("data/nnunet_results")

# Se crean los directorios (para las variables de entorno)
for path in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
    path.mkdir(parents=True, exist_ok=True)

# Variables de entorno nnU-Net
os.environ['nnUNet_raw'] = str(NNUNET_RAW)
os.environ['nnUNet_preprocessed'] = str(NNUNET_PREPROCESSED)
os.environ['nnUNet_results'] = str(NNUNET_RESULTS)

print("Variables de entorno configuradas:")
for var in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
    print(f"  {var}: {os.environ[var]}")

# ============================================================================
# FUNCIONES UTILIDADES
# ============================================================================

def ensure_dir(path: Path):
    """Crea directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)

def find_patient_dirs(base_path: Path):
    """Encuentra directorios de pacientes."""
    return sorted([p for p in base_path.iterdir() if p.is_dir()])

def get_modality_files(patient_dir: Path):
    """Identifica archivos de modalidades por nombre."""
    files = list(patient_dir.glob("*.nii.gz"))
    modality_files = {}

    for f in files:
        name = f.name.lower()
        if "seg" in name:
            modality_files["seg"] = f
        elif "flair" in name and "baseline" in name:
            modality_files["flair_baseline"] = f
        elif "flair" in name and "followup" in name:
            modality_files["flair_followup"] = f

    return modality_files

# ============================================================================
# 1. CREAMOS DATASET102_IMAGINEM_FLAIR (349 CASOS COMPLETOS)
# ============================================================================

def crear_dataset102():
    """Crea Dataset102 con todos los 349 casos."""
    print("\n" + "="*80)
    print("1. CREANDO DATASET102_IMAGINEM_FLAIR (349 CASOS)")
    print("="*80)

    dataset_name = "Dataset102_ImaginEM_FLAIR"
    dataset_dir = NNUNET_RAW / dataset_name

    imagesTr = dataset_dir / "imagesTr"
    labelsTr = dataset_dir / "labelsTr"
    ensure_dir(imagesTr)
    ensure_dir(labelsTr)

    # Configuración dataset.json, sólo dos canales, FLAIR baseline y FLAIR followup, para estar en consonancia con MSSEG2
    dataset_json = {
        "channel_names": {
            "0": "FLAIR_baseline",
            "1": "FLAIR_followup"
        },
        "labels": {
            "background": 0,
            "lesion_stable": 1,
            "lesion_new": 2
        },
        "numTraining": 0,  # El contador se actualiza
        "file_ending": ".nii.gz"
    }

    # Se procesan los IDs
    patient_dirs = find_patient_dirs(BASE_DIR)
    counter = 0

    MODALITY_MAP = {"flair_baseline": 0, "flair_followup": 1}
    required_keys = set(MODALITY_MAP.keys()) | {"seg"}

    for patient_dir in tqdm(patient_dirs, desc="Procesando pacientes"):
        pid = patient_dir.name
        modality_files = get_modality_files(patient_dir)

        if not required_keys.issubset(modality_files.keys()):
            print(f"⏭ Saltando {pid}: faltan archivos")
            continue

        case_id = f"ImaginEM_{pid}"

        # Se copian las imágenes (2 canales)
        for modality_name, channel_idx in MODALITY_MAP.items():
            src = modality_files[modality_name]
            dst = imagesTr / f"{case_id}_{channel_idx:04d}.nii.gz"
            shutil.copy(src, dst)

        # Copiar segmentación
        dst_label = labelsTr / f"{case_id}.nii.gz"
        shutil.copy(modality_files["seg"], dst_label)

        counter += 1
        print(f" Procesado {pid} → {case_id}")

    # Se actualiza dataset.json
    labels_tr_list = list(labelsTr.glob("*.nii.gz"))
    dataset_json["numTraining"] = len(labels_tr_list)

    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\n Dataset102 creado: {dataset_json['numTraining']} casos")
    print(f" Ubicación: {dataset_dir}")
    return dataset_dir

# ============================================================================
# 2. HOLD-OUT SPLIT (70/15/15) (70% train, 155 validation y 15% test)
# ============================================================================

def holdout_split(dataset_dir: Path):
    """Divide dataset en train(70%)/val(15%)/test(15%) con semilla 42."""
    print("\n" + "="*80)
    print("2. HOLD-OUT SPLIT (70/15/15) - SEMILLA 42")
    print("="*80)

    labelsTr = dataset_dir / "labelsTr"
    cases = sorted([f.name.replace(".nii.gz", "") for f in labelsTr.glob("*.nii.gz")])

    print(f"Total casos disponibles: {len(cases)}")

    # División reproducible
    random.seed(42) # semilla de reproducibilidad, la división de 70%, 15%, 15% se hace de forma aleatoria para evitar sesgos
    random.shuffle(cases)

    n = len(cases)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    # Cada conjunto es disjunto entre sí
    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    print(f"TRAIN: {len(train_cases)} casos ({len(train_cases)/n*100:.1f}%)")
    print(f" VAL:  {len(val_cases)} casos ({len(val_cases)/n*100:.1f}%)")
    print(f"TEST:  {len(test_cases)} casos ({len(test_cases)/n*100:.1f}%)")

    # Se guarda el split
    split_json = {
        "seed": 42,
        "total_cases": n,
        "train": train_cases,
        "val": val_cases,
        "test": test_cases
    }

    with open(dataset_dir / "split_holdout.json", "w") as f:
        json.dump(split_json, f, indent=4)

    print(f" split_holdout.json guardado")
    return train_cases, val_cases, test_cases

# ============================================================================
# 3. CREAR DATASET202_HOLDOUT (TRAIN+VAL)
# ============================================================================

def crear_dataset202(dataset102_dir: Path, train_cases, val_cases):
    """Crea Dataset202 con train+val para entrenamiento."""
    print("\n" + "="*80)
    print("3. CREANDO DATASET202_IMAGINEM_FLAIR_HOLDOUT (TRAIN+VAL)")
    print("="*80)

    dataset_name = "Dataset202_ImaginEM_FLAIR_HOLDOUT"
    src_dir = dataset102_dir
    dst_dir = NNUNET_RAW / dataset_name

    imagesTr_src = src_dir / "imagesTr"
    labelsTr_src = src_dir / "labelsTr"

    imagesTr_dst = dst_dir / "imagesTr"
    labelsTr_dst = dst_dir / "labelsTr"

    ensure_dir(imagesTr_dst)
    ensure_dir(labelsTr_dst)

    # Casos para entrenamiento (train + val)
    train_val_cases = set(train_cases + val_cases)

    for case_id in tqdm(train_val_cases, desc="Copiando train+val"):
        # Imágenes (2 canales)
        for ch in ["0000", "0001"]:
            src_img = imagesTr_src / f"{case_id}_{ch}.nii.gz"
            if src_img.exists():
                shutil.copy(src_img, imagesTr_dst / src_img.name)

        # Segmentación
        src_lbl = labelsTr_src / f"{case_id}.nii.gz"
        if src_lbl.exists():
            shutil.copy(src_lbl, labelsTr_dst / src_lbl.name)

    # Copiar y actualizar dataset.json
    with open(src_dir / "dataset.json") as f:
        ds_json = json.load(f)

    ds_json["numTraining"] = len(train_val_cases)

    with open(dst_dir / "dataset.json", "w") as f:
        json.dump(ds_json, f, indent=4)

    print(f" Dataset202 creado: {ds_json['numTraining']} casos (train+val)")
    print(f" Ubicación: {dst_dir}")
    return dst_dir

# ============================================================================
# 4. Se realiza el preprocesado con  nnU-Net v2
# ============================================================================

def preprocesar_dataset202():
    """Ejecuta nnUNetv2_plan_and_preprocess -d 202."""
    print("\n" + "="*80)
    print("4. PREPROCESADO nnUNetv2_plan_and_preprocess -d 202")
    print("="*80)

    cmd = "nnUNetv2_plan_and_preprocess -d 202"
    print(f"🛠️  Ejecutando: {cmd}")

    # EJECUTAR COMANDO (descomenta cuando tengas nnU-Net instalado)
    # os.system(cmd)

    preprocessed_path = Path(os.environ['nnUNet_preprocessed']) / "Dataset202_ImaginEM_FLAIR_HOLDOUT"

    if preprocessed_path.exists():
        configs = list(preprocessed_path.glob("nnUNetPlans*"))
        print(f" Preprocesado completado:")
        print(f"   Configuraciones: {[c.name for c in configs]}")
        print(f"    {preprocessed_path}")
    else:
        print("  Preprocesado no encontrado. Ejecuta manualmente:")
        print(f"   {cmd}")

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print(" Iniciando Holdout + Preprocesado nnU-Net")

    # 1. Dataset102 completo
    dataset102_dir = crear_dataset102()

    # 2. Holdout split
    train_cases, val_cases, test_cases = holdout_split(dataset102_dir)

    # 3. Dataset202 (train+val)
    dataset202_dir = crear_dataset202(dataset102_dir, train_cases, val_cases)

    # 4. Preprocesado
    preprocesar_dataset202()

    print("\n" + " HOLDOUT + PREPROCESADO FINALIZADO")
    print(f" split_holdout.json: {dataset102_dir / 'split_holdout.json'}")
    print(f" Dataset202 listo para entrenar: {dataset202_dir}")
    print(f" Preprocesado en: {NNUNET_PREPROCESSED / 'Dataset202_ImaginEM_FLAIR_HOLDOUT'}")
