import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import time
import SimpleITK as sitk
import os
import glob
import math
import tqdm 

# from numba import jit, prange
# @jit(nopython=True, nogil=True, cache=True, parallel=True, fastmath=True)
# def compute_tp_tn_fp_fn(y_true, y_pred):
#     tp = 0
#     tn = 0
#     fp = 0
#     fn = 0
#     for i in range(y_pred.size):
#         tp += y_true[i] * y_pred[i]
#         tn += (1-y_true[i]) * (1-y_pred[i])
#         fp += (1-y_true[i]) * y_pred[i]
#         fn += y_true[i] * (1-y_pred[i])


def compute_tp_tn_fp_fn(y_true, y_pred):
 
    tp = np.sum(y_true*y_pred)
    tn = np.sum((1-y_true)*(1-y_pred))
    fp = np.sum((1-y_true)*y_pred)
    fn = np.sum(y_true*(1-y_pred))

    return tp, tn, fp, fn

def compute_precision(tp, fp):
    return tp / (tp + fp)

def compute_recall(tp, fn):
    return tp / (tp + fn)

def compute_f1_score(precision, recall):
    try:
        return (2*precision*recall) / (precision + recall)
    except:
        return 0

def compute_fbeta_score(precision, recall, beta):
    try:
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    except:
        return 0

def compute_accuracy(tp,tn,fp,fn):
    return (tp + tn)/(tp + tn + fp + fn)

def compute_auc(GT, pred):
    return metrics.roc_auc_score(GT, pred)

def compute_auprc(GT, pred):
    prec, rec, thresholds = metrics.precision_recall_curve(GT, pred)
    # print(prec, rec, thresholds)
    plt.plot(prec, rec)
    plt.show()
    # return metrics.auc(prec, rec)

def compute_average_precision(GT, pred):
    ratio = sum(GT)/np.size(GT)
    return metrics.average_precision_score(GT, pred), ratio

dir = "/Users/luciacev-admin/Desktop/TEST_METRICS"
patients = {}
normpath = os.path.normpath("/".join([dir, '**', '']))
for img_fn in sorted(glob.iglob(normpath, recursive=True)):
    #  print(img_fn)
    basename = os.path.basename(img_fn)

    if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
        file_name = basename.split(".")[0]
        patient = file_name.split("_Pred_Sp")[0].split("_seg_Sp")[0].split("_scan_Sp")[0]

        # print(patient)

        if patient not in patients.keys():
            patients[patient] = {}

        if "_Pred_" in basename:
            patients[patient]["pred"] = img_fn

        elif "_seg_" in basename:
            patients[patient]["seg"] = img_fn
        # else:
        #     print("----> Unrecognise CBCT file found at :", img_fn)

# print(patients)

avg_recall = []
avg_precision = []
avg_f1 = []
avg_fbeta = []
avg_acc = []

metrics_names = ['AUPRC','AUPRC - Baseline','F1_Score','Fbeta_Score','Accuracy','Recall','Precision','File']
total_values = pd.DataFrame(columns=metrics_names)


startTime = time.time()


for patient, data in tqdm.tqdm(patients.items()):

    GT = sitk.ReadImage(data["seg"]) 
    GT = sitk.GetArrayFromImage(GT).flatten()

    pred = sitk.ReadImage(data["pred"]) 
    pred = sitk.GetArrayFromImage(pred).flatten()

    tp, tn, fp, fn = compute_tp_tn_fp_fn(GT,pred)
    recall = compute_recall(tp, fn)
    precision = compute_precision(tp, fp)
    f1 = compute_f1_score(precision, recall)
    fbeta = compute_fbeta_score(precision, recall, 2)
    acc = compute_accuracy(tp, tn, fp, fn)
    auprc, ratio = compute_average_precision(GT, pred)

    avg_recall.append(recall)
    avg_precision.append(precision)
    avg_f1.append(f1)
    avg_fbeta.append(fbeta)
    avg_acc.append(acc)

    # print("========================")
    # print(patient)
    # # print(tp,tn,fp,fn)
    # print("Recall",recall)
    # print("Precision",precision)
    # print("F1",f1)
    # print("Fbeta",fbeta)
    # print("Accuracy",acc)
    # print("========================")

    metrics_line = [auprc,ratio,f1,fbeta,acc,recall,precision]
    metrics_line.append(os.path.basename(data["pred"]).split('.')[0])
    total_values.loc[len(total_values)] = metrics_line

mean_values = pd.DataFrame(columns=metrics_names)

mean_line = []
std_line = []
for met_name in metrics_names[:-1]:
    means = total_values[met_name].mean()
    stds = total_values[met_name].std()
    mean_line.append(means)
    std_line.append(stds)

mean_line.append("Mean")
mean_values.loc[len(mean_values)] = mean_line

std_line.append("STD")
mean_values.loc[len(mean_values)] = std_line

endTime = time.time()

total_values.to_excel("All_metrics.xlsx")
mean_values.to_excel("Average_metrics.xlsx")

print(total_values)
print("Took",endTime-startTime,"s")
