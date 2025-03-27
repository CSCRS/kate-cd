from datasets import Dataset, DatasetDict, Image
import glob

def create_dataset(split):
        
    # Get sorted lists of file paths
    pre_image_paths = sorted(glob.glob(f"../data/label_studio_pre_post/{split}/A/*.png"))
    post_image_paths = sorted(glob.glob(f"../data/label_studio_pre_post/{split}/B/*.png"))
    label_paths = sorted(glob.glob(f"../data/label_studio_pre_post/{split}/label/*.png"))

    # Ensure all lists have the same length and filenames match
    assert len(pre_image_paths) == len(post_image_paths) == len(label_paths)

    # Create a Dataset
    dataset = Dataset.from_dict({
        "pre_image": pre_image_paths,
        "post_image": post_image_paths,
        "label": label_paths
    })

    # Cast columns to Image format
    dataset = dataset.cast_column("pre_image", Image())
    dataset = dataset.cast_column("post_image", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset

train_dataset = create_dataset("train")
val_dataset = create_dataset("val")
test_dataset = create_dataset("test")

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

from huggingface_hub import HfApi

# Create a new dataset repository on Hugging Face
api = HfApi()
#api.create_repo(repo_id="cscsr/kate-cd", repo_type="dataset")

# Push the dataset
dataset_dict.push_to_hub("cscsr/kate-cd")