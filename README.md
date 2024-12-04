# Brain_Tumor_Segmentation
This project uses deep learning to accurately detect and segment brain tumors from MRI images. Follow the steps below to set up and run the project.
## Steps to Build and Run the Project
### 1. Clone the Repository
- Open CMD or Terminal in the directory where you want to clone the repository.
- Run the following command:
  ``` bash
  git clone https://github.com/CodeWizardRakesh/Brain_Tumor_Segmentation.git
  ```
- Navigate to the cloned directory.
  ``` bash
  cd Brain_Tumor_Segmentation
  ```
### 2. Download and Prepare the Dataset
- Create a directory for the dataset
  ``` bash
  mkdir Dataset
  ```
- Download the dataset from [here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- Prepare the dataset by removing unnecessary files and organizing the data
  ``` bash
  del .\Dataset\kaggle_3m\README.md
  ```
  ``` bash
  mv .\Dataset\kaggle_3m\data.csv .\
  ```
### 3. Create a Conda Environment
- Open the Anaconda Prompt (download it if you haven't already).
- Run the following command to create a new environment with the necessary dependencies
  ```bash
  conda env create -f Environment_setup.yml --prefix ./env
  ```
### 4. Setup CUDA (Optional, for NVIDIA GPU Users)
- Follow the instructions in [this]() GitHub repo for easy installation.
  
