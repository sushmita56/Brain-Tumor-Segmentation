### 3D Brain Tumor Segmentation with PyTorch

This project implements a 3D U-Net model to perform brain tumor segmentation on the BraTS 2023 dataset. It includes a full pipeline for loading multi-modal MRI scans (T1n, T1c, T2w, T2f), preprocessing the data for efficiency, and training a 3D segmentation model.

Work done
- Data Preprocessing: Converted raw .nii.gz files into lightweight .npy files, which speeds up data loading during training.
- Data Augmentation: Used the Torchio library to apply random flips and affine transformations, helping the model generalize better and prevent overfitting.
- The project is organized into logical modules (model.py, dataset.py, train.py) for clarity and reusability.

Data Source: [Synapse](https://www.synapse.org/Synapse:syn51514105)
