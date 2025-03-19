import os
import numpy as np
from skimage import io
from scipy import stats
import argparse

def calc_TP(pred, label):

    pred = (pred >= 0.5)
    label = (label >= 0.5)
    
    TP = np.sum(pred & label)
    TN = np.sum(~pred & ~label)
    FP = np.sum(pred & ~label)
    FN = np.sum(~pred & label)

    return TP, TN, FP, FN


def calculate_metrics(preds, GTs):

    TP_total, TN_total, FP_total, FN_total = 0, 0, 0, 0

    for pred, gt in zip(preds, GTs):
        TP, TN, FP, FN = calc_TP(pred, gt)
        TP_total += TP
        TN_total += TN
        FP_total += FP
        FN_total += FN

    precision = TP_total / (TP_total + FP_total + 1e-10)
    recall = TP_total / (TP_total + FN_total + 1e-10)
    IoU0 = TP_total / (FP_total + TP_total + FN_total + 1e-10)
    IoU1 = TN_total / (FP_total + TN_total + FN_total + 1e-10)
    mIoU = (IoU0 + IoU1) / 2
    acc = (TP_total + TN_total) / (TP_total + FP_total + FN_total + TN_total + 1e-10)
    F1 = stats.hmean([precision, recall])

    return {
        "Precision": precision,
        "Recall": recall,
        "IoU0": IoU0,
        "IoU1": IoU1,
        "mIoU": mIoU,
        "Accuracy": acc,
        "F1 Score": F1
    }


def generate_summary_metrics(pred_resnets, pred_samcds, pred_effs, GTs, filename="summary.txt"):
    """
    ResNet, SAM-CD ve EffSAM için metrikleri hesaplar ve 'summary.txt' dosyasına kaydeder.
    """
    metrics_resnet = calculate_metrics(pred_resnets, GTs)
    metrics_samcd = calculate_metrics(pred_samcds, GTs)
    metrics_effsam = calculate_metrics(pred_effs, GTs)

    with open(filename, "w") as f:
        f.write("Summary of Model Performance\n")
        f.write("="*40 + "\n")

        for model_name, metrics in zip(["ResNet", "SAM-CD", "EffSAM"], [metrics_resnet, metrics_samcd, metrics_effsam]):
            f.write(f"\nModel: {model_name}\n")
            f.write("-"*40 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")

def calculate_average_iou(preds, GTs):
    iou_scores = []

    for pred_mask, gt_mask in zip(preds, GTs):
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)

        iou_score = (np.sum(intersection) + 1e-7) / (np.sum(union) + 1e-7)
        
        iou_scores.append(iou_score)  

    average_iou = np.mean(iou_scores) if iou_scores else 0.0
    return average_iou, iou_scores  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--partition", help="train, test or val", required=True)
    args = parser.parse_args()

    partition = args.partition

    pre_dir = f'../data/label_studio_pre_post/{partition}/A'
    post_dir = f'../data/label_studio_pre_post/{partition}/B'
    GT_dir = f'../data/label_studio_pre_post/{partition}/label'
    
    resnet_dir = f'../results/{partition}/ResNet'
    samcd_dir = f'../results/{partition}/SAM'
    effsam_dir = f'../results/{partition}/effSAM'
    
    image_name_list = [f for f in os.listdir(pre_dir) if f.endswith('.png')]
    GTs, pred_resnets, pred_samcds, pred_effs = [], [], [], []

    for it in image_name_list:
        gt = io.imread(os.path.join(GT_dir, it)) // 255
        GTs.append(gt)
        
        pred_resnet = io.imread(os.path.join(resnet_dir, it)) // 255
        pred_resnets.append(pred_resnet)
        
        pred_samcd = io.imread(os.path.join(samcd_dir, it)) // 255
        pred_samcds.append(pred_samcd)
        
        pred_effsam = io.imread(os.path.join(effsam_dir, it)) // 255
        pred_effs.append(pred_effsam)


    generate_summary_metrics(pred_resnets, pred_samcds, pred_effs, GTs, filename=f"../results/{partition}_scores.txt")
    print(f"Summary metrics saved to ../results/{partition}_scores.txt")