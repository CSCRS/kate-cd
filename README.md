# Welcome to the KATE-CD Dataset
Welcome to the home page of Kahramanmaraş Türkiye Earthquake-Change Detection Dataset (KATE-CD). If you are reading this README, you are probably visiting one of the following places to learn more about **KATE-CD Dataset** and the associated study "*Earthquake Damage Assessment with SAMCD: A Change Detection Approach for VHR Images*", to be presented in 17th International Conference on Joint Urban Remote Sensing ([JURSE 2025](https://2025.ieee-jurse.org)). 

* [Code Ocean Capsule](https://doi.org/10.24433/CO.3747729.v1) in [Open Science Library](https://codeocean.com/explore/82765786-a936-438c-a75a-84e2817294c5) 
* [GitHub Repository](https://github.com/cscrs/kate-cd)
* [HuggingFace](https://huggingface.co/datasets/cscrs/kate-cd)

The **KATE-CD Dataset** and the associated codes and supplementary information are published in three places i.e. CodeOcean, GitHub and HuggingFAce for providing redundancy and extended reach. All of the content uploaded to the three websites are the same except small differences because of platform requirements. The CodeOcean platform is mainly used for reproducibility, whereas GitHub is used to provide git access and hence easy collaboration between **KATE-CD Dataset** developers. Finally HuggingFace provides an easy access to the database where you can run existing models in HuggingFace on KATE-CD without too much effort.

## Content of the KATE-CD Dataset
The KATE-CD dataset is designed to facilitate the development and evaluation of change detection algorithms for earthquake damage assessment. It provides high-resolution, bitemporal satellite imagery from pre- and post-earthquake events, specifically covering the regions affected by the Kahramanmaraş earthquake in Türkiye. Given the scarcity of earthquake damage assessment datasets, KATE-CD aims to bridge this gap by offering high-quality annotated data, enabling researchers to train and test machine learning models for automated damage detection.

### Source of Satellite Imagery
The dataset includes satellite images from Maxar Open Data and Airbus Pleiades, covering seven heavily affected cities: Adıyaman, Gaziantep, Hatay, Kahramanmaraş, Kilis, Osmaniye, and Malatya. These images have a resolution ranging from 0.3m to 0.5m. The collection process involved selecting imagery captured under various lighting conditions, using different sensors and viewing angles. The coordinate reference system EPSG:32637 was chosen for consistency, and radiometrically corrected images with 8-bit spectral resolution were used to maintain uniform color representation across sources.

### Labelling Process
A grid-based labeling approach was used to divide the images into 512×512 pixel patches. The Label Studio tool was employed for manual annotation, where 834 post-earthquake images were reviewed, and damaged buildings were marked with polygonal annotations. Each labeled image was then paired with its corresponding pre-earthquake image, resulting in 486 pre/post image pairs for change detection. A binary labeling strategy was applied, where pixels inside damage polygons were assigned a value of 1 (damaged), and all others were set to 0 (undamaged).

### Machine-Learning Ready Format
To integrate with the change detection frameworks, the dataset was structured into a standardized format. This included organizing image pairs into directories suitable for model training. The dataset was made publicly available on CodeOcean, GitHub, and HuggingFace, allowing reproducibility and accessibility for researchers.

## Reproducibility: CodeOcean
The dataset is published on three platforms: CodeOcean, GitHub and HuggingFace. The purpose of CodeOcean is to provide data, codes and the computing instructions to reproduce the results. CodeOcean uses the term *capsule* to define the collection of everything needed to reproduce the results. Depending on your goal and your time constraints, CodeOcean provide two alternatives to run the capsule and obtain the results: via Open Science Library or Capsule Export.

### Open Science Library
If you visit [this capsule](https://doi.org/10.24433/CO.3747729.v1) via [Open Science Library](https://codeocean.com/explore/82765786-a936-438c-a75a-84e2817294c5) developed by [Code Ocean](https://codeocean.com), then you should be able to see the published results in the results folder of the capsule. Code Ocean has an internal publishing process to verify that on each run the capsule will produce the same results. So, if you are in a hurry, or don't bother running the capsule again, then you can take a look at the published results and check the codes and data in the capsule.

If you want to run the capsule and produce results by yourself, then all you have to do is to click "Reproducible Run" button in the capsule page. The [Open Science Library](https://codeocean.com/explore/82765786-a936-438c-a75a-84e2817294c5) will run the capsule from the scratch and produce the results.

### Capsule Export
If you would like to use your own computing resources for reproduction, then you can export the capsule via "Capsule"--> "Export" menu to your working environment. Please make sure to check "Include Data" option while exporting. After extracting the export file, you should follow the instructions in "REPRODUCING.md". For the sake of completeness, we mention the procesures here.

#### Prerequisites 
- [Docker Community Edition (CE)](https://www.docker.com/community-edition)

#### Building the environment
This capsule has been published and its environment has been archived and made available on Code Ocean's Docker registry:
`registry.codeocean.com/published/82765786-a936-438c-a75a-84e2817294c5:v1`

### Running the capsule to reproduce the results
In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm --gpus all \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  registry.codeocean.com/published/82765786-a936-438c-a75a-84e2817294c5:v1 bash run
```

## Published results
In the results folder of the CodeOcean capsule, you can reach the pre-computed outputs of the code or you can generate them from scratch with single-click in CodeOcean. In either case, these outputs correspond to the published content in the manuscript. The mapping between capsule results and the content in the manuscript is as follows:

    Code              Outputs           Manuscript
    ------------      ----------------  ----------
    predictions.py    val_scores.txt    Table II
    evaluate.py       train_scores.txt
                      test_scores.txt 
    ---------------------------------------------
    visualization.py  val_plots.pdf     Figure 2
                      train_plots.pdf
                      test_plots.pdf
    ---------------------------------------------
                       

## For Developers
### Differences between the platforms:
* CodeOcean is the primary source of the dataset (*data* folder) and the codes (*code* folder).
* GitHub does not contain *data* folder because GitHub is not designed to store and manage large files.
* [HuggingFace dataset](https://huggingface.co/datasets/cscrs/kate-cd) is hosted on an isolated repository with Large File Support (LFS). In this isolated repo, Parquet files of the original *data* folder are served. It also includes a README file with an auto-generated metadata for visual presentation on HuggingFace.
  
### GitHub Access
If you would like to look at the capsule more closely, and build a working development environment then you can use [Development Containers](https://containers.dev/) functionality of VSCode. For this purpose, we created **.devcontainer/devcontainer.json** file under the root folder in the capsule. The purpose of this config file is to tell VSCode the location of the **Dockerfile**. Here, for design purposes, we used the same **Dockerfile** provided by CodeOcean. In this way, we do not interfere the building process in the CodeOcean. 

To open the GitHub repository in DevContainers, you can click the button below. It will open the VSCode in DevContainers mode and fetch the GitHub repository automatically.

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/cscrs/kate-cd.git)

### CodeOcean Access
To open the capsule in VSCode via [Development Containers](https://containers.dev/), you first need to download the capsule. There are two ways: either you can use the capsule export, or you can pull the capsule from git repository. We recommend using git. You can use either CodeOcean or GitHub repository (both have the same content).

~~~bash
# CodeOcean git repository
# There are two git repos in CodeOcean. 
# (1) Repo of the published capsule (https://git.codeocean.com/capsule-9061546.git)
# (2) Repo of the original capsule  (https://git.codeocean.com/capsule-3747729.git)
# Here, we are using the git repo of the original capsule
$ git clone https://git.codeocean.com/capsule-3747729.git
~~~
or 
~~~bash
# GitHub git repository
$ git clone https://github.com/cscrs/kate-cd.git
~~~

The next step is to open VSCode, select *Open a Remote Window* and then *Open Folder in Container..." option. Select your cloned git folder and the VSCode should start building Docker container and open the content of the capsule. 

### HuggingFace Access
[HuggingFace](https://huggingface.co/datasets/cscrs/kate-cd) is used to host the database and provide a nice visual access to the developers. HuggingFace uses the Parquet format to host the database. HuggingFace also uses Large File Support (LFS) for the internal git repository. Therefore, we decided to isolate the git repository of HuggingFace from GitHub and CodeOcean. The git repository of HuggingFace host only the database (in Parquet format) and a README. 

The Parquet files in the HuggingFace repository are updated via:

    $ cd code
    $ python utils/hf_update_db.py

### Relative vs Absolute paths
We use relative paths to locate the data files in the code to achieve compatibility between different working environments. In this way, the same codes and data structure can be used without any change if one tries to run the capsule on [Open Science Library](https://codeocean.com/explore/82765786-a936-438c-a75a-84e2817294c5) or local development environment. 

The only requirement of relative-path approach is to run Python codes within the **code** folder similar to this:
~~~bash
$ cd code
$ python predictions.py
~~~
This approach also fits to the way how CodeOcean runs the capsule.

### Reproducibility results folder
If you visit [Open Science Library](https://codeocean.com/explore/82765786-a936-438c-a75a-84e2817294c5), you will see that published results are always populated under **results** folder. This is a special folder CodeOcean uses to store the outputs likes PDFs, PNGs, or ordinary text outputs. Therefore, in CodeOcean capsules **results** folder is not included in *git* structure. So, when you pull or export a CodeOcean capsule, you won't see this folder. Whenever you create an output, you should create **results** folder and put the outputs under it. For the same reason, you should not include it to git. 




 
