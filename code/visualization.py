import argparse
import numpy as np
from skimage import io, measure
from scipy import stats
from utils.metric_tool import get_mIoU
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from utils.utils import align_dims
import os




def calc_conf_matrix(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred>= 0.5)
    label = (label>= 0.5)

    h, w = label.shape
    image = np.zeros((h, w, 3))
    GT = label
    TP = (pred * label).astype(int)
    FP = (pred * (~label)).astype(int)
    FN = ((~pred) * (label)).astype(int)
    TN = ((~pred) * (~label)).astype(int)

    image[TP == 1] = [1, 1, 1]  
    image[TN == 1] = [0, 0, 0]  
    image[FP == 1] = [1, 0, 0]  
    image[FN == 1] = [0, 0, 1] 
    return image

def calculate_average_iou(preds, GTs):
    iou_scores = []

    for pred_mask, gt_mask in zip(preds, GTs):
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)

        iou_score = (np.sum(intersection) + 1e-7) / (np.sum(union) + 1e-7)
        
        iou_scores.append(iou_score)  

    average_iou = np.mean(iou_scores) if iou_scores else 0.0
    return average_iou, iou_scores  

def save_images_as_pdf(pre, post, gt, pred_resnet, pred_samcd, pred_effs, iou_scores, image_name_list, filename="output.pdf"):

    sorted_indices = sorted(range(len(iou_scores)), key=lambda i: iou_scores[i], reverse=True)
    
    pre_sorted = [pre[i] for i in sorted_indices]
    post_sorted = [post[i] for i in sorted_indices]
    gt_sorted = [gt[i] for i in sorted_indices]
    pred_resnet_sorted = [pred_resnet[i] for i in sorted_indices]
    pred_samcd_sorted = [pred_samcd[i] for i in sorted_indices]
    pred_effs_sorted = [pred_effs[i] for i in sorted_indices]
    iou_sorted = [iou_scores[i] for i in sorted_indices]
    image_name_list = [image_name_list[i] for i in sorted_indices]

    num_images = min(12, len(image_name_list)) 
    fig, axes = plt.subplots(6, num_images, figsize=(3 * num_images, 18))  
    row_labels = ["Pre", "Post", "GT", "Resnet", "Fast SAM", "Efficient SAM"]
    column_labels = image_name_list[:num_images]

    for i in range(num_images):
        axes[0, i].imshow(pre_sorted[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(post_sorted[i])
        axes[1, i].axis("off")

        axes[2, i].imshow(gt_sorted[i], cmap="gray")
        axes[2, i].axis("off")

        axes[3, i].imshow(pred_resnet_sorted[i])
        axes[3, i].axis("off")

        axes[4, i].imshow(pred_samcd_sorted[i])
        axes[4, i].axis("off")

        axes[5, i].imshow(pred_effs_sorted[i])
        axes[5, i].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, 
            xy=(-0.1, 0.5), 
            xycoords="axes fraction", 
            fontsize=24, 
            fontweight="normal", 
            rotation=0, 
            ha="right", 
            va="center"
        )

    for col, label in enumerate(column_labels[:num_images]):
        axes[0, col].annotate(
            label, 
            xy=(0.5, 1.05), 
            xycoords="axes fraction", 
            fontsize=24, 
            fontweight="normal", 
            rotation=0, 
            ha="center", 
            va="bottom"
        )

    legend_labels = ["True Negative", "True Positive", "False Positive", "False Negative"]
    legend_colors = ["black", "white", "red", "blue"]

    patches = [mlines.Line2D([], [], color=legend_colors[i], marker='s', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5, linestyle='None', 
                label=legend_labels[i]) for i in range(len(legend_labels))]

    fig.legend(
        handles=patches, 
        bbox_to_anchor=(0.5, 0.05), 
        loc="lower center",  
        ncol=4,  
        fontsize=24,  
        frameon=True,
        markerscale=2 
    )

    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.1, wspace=0.1)
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()


def main():

    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("--partition", help="train, test or val", required=True)
    args = parser.parse_args()
    partition = args.partition


    pre_dir = f"../data/label_studio_pre_post/{partition}/A"
    post_dir = f"../data/label_studio_pre_post/{partition}/B"
    GT_dir = f"../data/label_studio_pre_post/{partition}/label"
    
    resnet_dir = f"../results/{partition}/ResNet"
    samcd_dir = f"../results/{partition}/SAM"
    effsam_dir = f"../results/{partition}/effSAM"

    image_name_list = [f for f in os.listdir(pre_dir) if f.endswith('.png')]
    pres, posts, GTs = [], [], []
    pred_resnets, pred_samcds, pred_effs = [], [], []
    iou_score_pred = []

    for it in image_name_list:
        pre = io.imread(os.path.join(pre_dir, it))
        post = io.imread(os.path.join(post_dir, it))
        pres.append(pre)
        posts.append(post)        
        gt = io.imread(os.path.join(GT_dir, it))
        gt = gt // 255 
        GTs.append(gt)
        
        pred_resnet = io.imread(os.path.join(resnet_dir, it)) // 255
        conf_pred_resnet = calc_conf_matrix(pred_resnet, gt)
        pred_resnets.append(conf_pred_resnet)
        
        pred_samcd = io.imread(os.path.join(samcd_dir, it)) // 255
        conf_pred_samcd = calc_conf_matrix(pred_samcd, gt)
        pred_samcds.append(conf_pred_samcd)
        
        pred_effsam = io.imread(os.path.join(effsam_dir, it)) // 255
        conf_pred_effsam = calc_conf_matrix(pred_effsam, gt)
        iou_score_pred.append(pred_effsam)

        pred_effs.append(conf_pred_effsam)

    average_iou, iou_scores = calculate_average_iou(iou_score_pred, GTs)
    save_images_as_pdf(pres, posts, GTs, pred_resnets, pred_samcds, pred_effs, iou_scores, image_name_list, f"../results/{partition}_plots.pdf")


if __name__ == '__main__':
    print(f'Plot generation started')
    main()
    print(f'Plot generation finished')



