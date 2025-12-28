#!/usr/bin/env python3
"""
Evaluación Métricas Completa - TFM Detección Lesiones Esclerosis Múltiple
========================================================================
Fase 05: Métricas voxel/lesion/patient-wise en ImaginEM test + MSSEG2

Modelos: 20/50/100/250 épocas
Datasets: Dataset210 (ImaginEM test 15%) + Dataset302 (MSSEG2 40 casos)

Métricas: Dice, Precision, Recall, F1 (estable/nueva), Sensitivity, etc.
Salida: 6 csv en outputs/05_metricas/
"""

import os
import glob
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from scipy.ndimage import label as cc_label

# Configuración matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

NNUNET_RAW = Path("data/nnunet_raw")
NNUNET_RESULTS = Path("data/nnunet_results")
OUTPUT_DIR = Path("outputs/05_metricas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variables de entorno
os.environ['nnUNet_raw'] = str(NNUNET_RAW)
os.environ['nnUNet_results'] = str(NNUNET_RESULTS)

print("Configuración:")
print(f"  NNUNET_RAW: {NNUNET_RAW}")
print(f"  NNUNET_RESULTS: {NNUNET_RESULTS}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")

# ============================================================================
# FUNCIONES EVALUACIÓN IMAGINEM (labels 0,1,2)
# ============================================================================

def eval_voxel_level_imaginem(preds_path, labels_path):
    """Voxel-wise: sólo voxeles con lesión en GT (1 vs 2)."""
    label_files = sorted(glob.glob(f"{labels_path}/*.nii.gz"))
    all_true_lesions = []
    all_pred_lesions = []

    for lf in label_files:
        name = os.path.basename(lf)
        pf = os.path.join(preds_path, name)
        if not os.path.exists(pf):
            print(f" Predicción faltante: {name}")
            continue

        y_true = nib.load(lf).get_fdata().astype(int).flatten()
        y_pred = nib.load(pf).get_fdata().astype(int).flatten()

        mask_lesion = y_true > 0
        all_true_lesions.extend(y_true[mask_lesion])
        all_pred_lesions.extend(y_pred[mask_lesion])

    all_true_lesions = np.array(all_true_lesions)
    all_pred_lesions = np.array(all_pred_lesions)

    report = classification_report(
        all_true_lesions, all_pred_lesions, labels=[1, 2],
        target_names=["lesion_stable", "lesion_new"], digits=4,
        output_dict=True, zero_division=0
    )

    cm = confusion_matrix(all_true_lesions, all_pred_lesions, labels=[1, 2])
    tn, fp, fn, tp = cm.ravel()
    dice_new = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "dice_new": dice_new,
        "precision_new": report["lesion_new"]["precision"],
        "recall_new": report["lesion_new"]["recall"],
        "f1_new": report["lesion_new"]["f1-score"],
        "precision_stable": report["lesion_stable"]["precision"],
        "recall_stable": report["lesion_stable"]["recall"],
        "f1_stable": report["lesion_stable"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }

def eval_lesion_level_imaginem(preds_path, labels_path, min_overlap_vox=1):
    """Lesion-wise: componentes conectadas label=2."""
    label_files = sorted(glob.glob(f"{labels_path}/*.nii.gz"))
    total_GT_lesions = total_detected_GT = total_FP_lesions = 0
    struct = np.ones((3, 3, 3), dtype=bool)

    for lf in label_files:
        name = os.path.basename(lf)
        pf = os.path.join(preds_path, name)
        if not os.path.exists(pf): continue

        y_true = nib.load(lf).get_fdata().astype(int)
        y_pred = nib.load(pf).get_fdata().astype(int)

        gt_bin = (y_true == 2)
        pred_bin = (y_pred == 2)

        gt_cc, n_gt = cc_label(gt_bin, structure=struct)
        pr_cc, n_pr = cc_label(pred_bin, structure=struct)

        # GT detectadas
        gt_detected = sum(1 for i in range(1, n_gt + 1)
                         if (pr_cc[gt_cc == i] > 0).sum() >= min_overlap_vox)

        # FP
        fp_lesions = sum(1 for j in range(1, n_pr + 1)
                        if (gt_cc[pr_cc == j] > 0).sum() < min_overlap_vox)

        total_GT_lesions += n_gt
        total_detected_GT += gt_detected
        total_FP_lesions += fp_lesions

    sens_lesion = total_detected_GT / total_GT_lesions if total_GT_lesions > 0 else 0.0
    prec_lesion = total_detected_GT / (total_detected_GT + total_FP_lesions) if (total_detected_GT + total_FP_lesions) > 0 else 0.0

    return {
        "total_GT_lesions": total_GT_lesions,
        "total_detected_GT": total_detected_GT,
        "total_FP_lesions": total_FP_lesions,
        "sensitivity_lesion": sens_lesion,
        "precision_lesion": prec_lesion,
    }

def eval_patient_level_imaginem(preds_path, labels_path):
    """Patient-wise: detecta si paciente tiene >=1 voxel label=2."""
    label_files = sorted(glob.glob(f"{labels_path}/*.nii.gz"))
    n_cases = n_cases_with_gt_new = n_cases_detected = 0

    for lf in label_files:
        name = os.path.basename(lf)
        pf = os.path.join(preds_path, name)
        if not os.path.exists(pf): continue

        y_true = nib.load(lf).get_fdata().astype(int)
        y_pred = nib.load(pf).get_fdata().astype(int)

        gt_has_new = (y_true == 2).any()
        pred_has_new = (y_pred == 2).any()

        n_cases += 1
        if gt_has_new:
            n_cases_with_gt_new += 1
            if pred_has_new:
                n_cases_detected += 1

    sens_case = n_cases_detected / n_cases_with_gt_new if n_cases_with_gt_new > 0 else 0.0
    return {
        "n_cases": n_cases,
        "n_cases_with_gt_new": n_cases_with_gt_new,
        "n_cases_detected": n_cases_detected,
        "sensitivity_case": sens_case,
    }

# ============================================================================
# FUNCIONES EVALUACIÓN MSSEG2 (labels 0,1, la predicción mapea 2 a 1)
# ============================================================================

def eval_msseg2_voxel(preds_path, labels_path):
    """Voxel-wise binario: GT(0,1) vs pred(2→1)."""
    label_files = sorted(glob.glob(f"{labels_path}/*.nii.gz"))
    TP = FP = FN = TN = 0

    for lf in label_files:
        name = os.path.basename(lf)
        pf = os.path.join(preds_path, name)
        if not os.path.exists(pf): continue

        y_true = nib.load(lf).get_fdata().astype(int).flatten()
        y_pred = (nib.load(pf).get_fdata().astype(int) == 2).astype(int).flatten()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        TN += tn; FP += fp; FN += fn; TP += tp

    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    prec_new = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec_new = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_new = 2 * prec_new * rec_new / (prec_new + rec_new) if (prec_new + rec_new) > 0 else 0.0

    return {
        "tp": int(TP), "fp": int(FP), "fn": int(FN), "tn": int(TN),
        "dice_new": dice, "precision_new": prec_new, "recall_new": rec_new, "f1_new": f1_new
    }

def eval_msseg2_lesion(preds_path, labels_path, min_overlap_vox=1):
    """Lesion-wise: GT=1 vs pred=2."""
    label_files = sorted(glob.glob(f"{labels_path}/*.nii.gz"))
    total_GT_lesions = total_detected_GT = total_FP_lesions = 0
    struct = np.ones((3, 3, 3), dtype=bool)

    for lf in label_files:
        name = os.path.basename(lf)
        pf = os.path.join(preds_path, name)
        if not os.path.exists(pf): continue

        y_true = nib.load(lf).get_fdata().astype(int)
        y_pred = nib.load(pf).get_fdata().astype(int)

        gt_bin = (y_true == 1)
        pred_bin = (y_pred == 2)

        gt_cc, n_gt = cc_label(gt_bin, structure=struct)
        pr_cc, n_pr = cc_label(pred_bin, structure=struct)

        gt_detected = sum(1 for i in range(1, n_gt + 1)
                         if (pr_cc[gt_cc == i] > 0).sum() >= min_overlap_vox)
        fp_lesions = sum(1 for j in range(1, n_pr + 1)
                        if (gt_cc[pr_cc == j] > 0).sum() < min_overlap_vox)

        total_GT_lesions += n_gt
        total_detected_GT += gt_detected
        total_FP_lesions += fp_lesions

    sens_lesion = total_detected_GT / total_GT_lesions if total_GT_lesions > 0 else 0.0
    prec_lesion = total_detected_GT / (total_detected_GT + total_FP_lesions) if (total_detected_GT + total_FP_lesions) > 0 else 0.0

    return {
        "total_GT_lesions": total_GT_lesions,
        "total_detected_GT": total_detected_GT,
        "total_FP_lesions": total_FP_lesions,
        "sensitivity_lesion": sens_lesion,
        "precision_lesion": prec_lesion,
    }

def eval_msseg2_patient(preds_path, labels_path):
    """Patient-wise: GT=1 vs pred=2."""
    label_files = sorted(glob.glob(f"{labels_path}/*.nii.gz"))
    n_cases = n_cases_with_gt_new = n_cases_detected = 0

    for lf in label_files:
        name = os.path.basename(lf)
        pf = os.path.join(preds_path, name)
        if not os.path.exists(pf): continue

        y_true = nib.load(lf).get_fdata().astype(int)
        y_pred = nib.load(pf).get_fdata().astype(int)

        gt_has_new = (y_true == 1).any()
        pred_has_new = (y_pred == 2).any()

        n_cases += 1
        if gt_has_new:
            n_cases_with_gt_new += 1
            if pred_has_new:
                n_cases_detected += 1

    sens_case = n_cases_detected / n_cases_with_gt_new if n_cases_with_gt_new > 0 else 0.0
    return {
        "n_cases": n_cases,
        "n_cases_with_gt_new": n_cases_with_gt_new,
        "n_cases_detected": n_cases_detected,
        "sensitivity_case": sens_case,
    }

# ============================================================================
# EVALUACIÓN PRINCIPAL
# ============================================================================

def evaluar_todos_modelos():
    """Evalúa 4 modelos en 2 datasets × 3 niveles = 24 métricas."""
    print("\n" + "="*80)
    print("EVALUACIÓN COMPLETA: 4 MODELOS × 2 DATASETS × 3 NIVELES")
    print("="*80)

    # Rutas datasets test
    labels_imaginem = str(NNUNET_RAW / "Dataset210_ImaginEM_FLAIR_HOLDOUT_TEST/labelsTs")
    labels_msseg2 = str(NNUNET_RAW / "Dataset302_MSSEG2_FLAIR/labelsTs")

    # Experimentos (20/50/100/250 epochs)
    experiments_imaginem = {
        20: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/test_holdout_preds_20ep"),
        50: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/test_holdout_preds_50ep"),
        100: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/test_holdout_preds_100ep"),
        250: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/test_holdout_preds_250ep"),
    }

    experiments_msseg2 = {
        20: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/msseg2_preds_20ep"),
        50: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/msseg2_preds_50ep"),
        100: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/msseg2_preds_100ep"),
        250: str(NNUNET_RESULTS / "Dataset202_ImaginEM_FLAIR_HOLDOUT/msseg2_preds_250ep"),
    }

    # IMAGINEM TEST
    print("\n EVALUANDO IMAGINEM TEST HOLDOUT (15%)")
    results_imag = {"voxel": [], "lesion": [], "patient": []}

    for epochs, preds_path in experiments_imaginem.items():
        print(f"\n--- Modelo {epochs} epochs ---")

        results_imag["voxel"].append({**eval_voxel_level_imaginem(preds_path, labels_imaginem), "epochs": epochs})
        results_imag["lesion"].append({**eval_lesion_level_imaginem(preds_path, labels_imaginem), "epochs": epochs})
        results_imag["patient"].append({**eval_patient_level_imaginem(preds_path, labels_imaginem), "epochs": epochs})

    # MSSEG2
    print("\n EVALUANDO MSSEG2 (40 casos)")
    results_msseg2 = {"voxel": [], "lesion": [], "patient": []}

    for epochs, preds_path in experiments_msseg2.items():
        print(f"\n--- Modelo {epochs} epochs ---")

        results_msseg2["voxel"].append({**eval_msseg2_voxel(preds_path, labels_msseg2), "epochs": epochs})
        results_msseg2["lesion"].append({**eval_msseg2_lesion(preds_path, labels_msseg2), "epochs": epochs})
        results_msseg2["patient"].append({**eval_msseg2_patient(preds_path, labels_msseg2), "epochs": epochs})

    # Guardamos los csv
    for level in ["voxel", "lesion", "patient"]:
        pd.DataFrame(results_imag[level]).sort_values("epochs").to_csv(OUTPUT_DIR / f"imaginem_{level}_metrics.csv", index=False)
        pd.DataFrame(results_msseg2[level]).sort_values("epochs").to_csv(OUTPUT_DIR / f"msseg2_{level}_metrics.csv", index=False)

    return results_imag, results_msseg2

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(" Fase 05: Evaluación Métricas Completa")

    results_imag, results_msseg2 = evaluar_todos_modelos()

    print("\n" + " EVALUACIÓN FINALIZADA")
    print(f"\n RESULTADOS GUARDADOS en {OUTPUT_DIR}:")
    print("   - imagin_em_voxel_metrics.csv")
    print("   - imagin_em_lesion_metrics.csv")
    print("   - imagin_em_patient_metrics.csv")
    print("   - msseg2_voxel_metrics.csv")
    print("   - msseg2_lesion_metrics.csv")
    print("   - msseg2_patient_metrics.csv")

    print("\n TABLAS RESUMEN:")
    for dataset, results in [("IMAGINEM", results_imag), ("MSSEG2", results_msseg2)]:
        print(f"\n{dataset} VOXEL-WISE:")
        print(pd.DataFrame(results["voxel"]).round(4)[["epochs", "dice_new", "f1_new"]])
