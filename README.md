[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/hkayabilisim/jurse2025.git)

This Open Science Library capsule contains codes, data and computing instructions to reproduce the results in "*Earthquake Damage Assessment with SAMCD: A Change Detection Approach for VHR Images*", the study presented in 17th International Conference on Joint Urban Remote Sensing ([JURSE 2025](https://2025.ieee-jurse.org)). 

# How to reproduce the results
Everything needed is already contained in this capsule. Depending on your goal and your time constraints, we provide two alternatives to run the capsule and obtain the results: via Open Science Library or Capsule Export.

## Open Science Library
If you visit this capsule via [Open Science Library](https://codeocean.com/explore) developed by [Code Ocean](https://codeocean.com), then you should be able to see the published results in the library. Code Ocean has an internal publishing process to verify that on each run the capsule will produce the same results. So, if you are in a hurry, or don't bother running the capsule again, then you can take a look at the published results and check the codes and data in the capsule.

If you want to run the capsule and produce results by yourself, then all you have to do is to click "Reproducible Run" button in the capsule page. The [Open Science Library](https://codeocean.com/explore) will run the capsule from the scratch and produce the results.

## Capsule Export
If you would like to use your own computing resources for reproduction, then you can export the capsule via "Capsule"--> "Export" menu to your working environment. Please make sure to check "Include Data" option while exporting. After extracting the export file, you should follow the instructions in "REPRODUCING.md". For the sake of completeness, we mention the procesures here.

### Prerequisites 
- [Docker Community Edition (CE)](https://www.docker.com/community-edition)

### The computational environment (Docker image)
In your terminal, navigate to the folder where you've extracted the capsule and execute the following command:
```shell
cd environment && docker build . --tag e24e05d9-fd7f-4584-878b-4f19e31b750c; cd ..
```

This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build.

### Running the capsule to reproduce the results
In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  e24e05d9-fd7f-4584-878b-4f19e31b750c bash run
```

# For Developers
If you would like to look at the capsule more closely, and build a working development environment then you can use [Development Containers](https://containers.dev/) functionality of VSCode. For this purpose, we created **.devcontainer/devcontainer.json** file under the root folder in the capsule. The purpose of this config file is to tell VSCode the location of the **Dockerfile**. Here, for design purposes, we used the same **Dockerfile** provided by CodeOcean. In this way, we do not interfere the building process in the CodeOcean. 

## Obtain the capsule
To open the capsule in VSCode via [Development Containers](https://containers.dev/), you first need to download the capsule. There are two ways: either you can use the capsule export, or you can pull the capsule from git repository. We recommend using git.

~~~bash
$ git clone https://git.codeocean.com/capsule-3747729.git
~~~

The next step is to open VSCode, select *Open a Remote Window* and then *Open Folder in Container..." option. Select your git folder and the VSCode should start building Docker container and open thhe content of the capsule. 


## Relative vs Absolute paths
We use relative paths to locate the data files in the code to achieve compatibility between different working environments. In this way, the same codes and data structure can be used without any change if one tries to run the capsule on [Open Science Library](https://codeocean.com/explore) or local development environment. 

The only requirement of relative-path approach is to run Python codes within the **code** folder similar to this:
~~~bash
$ cd code
$ python train.py
~~~
This approach also fits to the way how CodeOcean runs the capsule.

## results folder
When you visit [Open Science Library](https://codeocean.com/explore), you will see that published results are always populated under **results** folder. This is a special folder CodeOcean uses to store the outputs likes PDFs, PNGs, or ordinary text outputs. Therefore, in CodeOcean capsules **results** folder is not included in *git* structure. So, when you pull or export a CodeOcean capsule, you won't see this folder. Whenever you create an output, you should create **results** folder and put the outputs under it. For the same reason, you should not include it to git. 

# Results
The mapping between capsule results and the content in the manuscript is as follows:

    Code            CodeOcean         Manuscript
    ------------    ----------------  ----------
    predictions.py  val_scores.txt    Table II
    evaluate.py     train_scores.txt
                    test_scores.txt 
    ---------------------------------------------

For the explanation of the simulations, please check the related code.

 
