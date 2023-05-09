# MedMNIST Experiments 

This repository contains the code for the experiments of [MedMNIST](https://github.com/MedMNIST/MedMNIST/). 
Most of the code presented here is based on the official MedMNIST code and official LibAUC code.

- https://github.com/MedMNIST/experiments
- https://github.com/MedMNIST/MedMNIST
- https://github.com/Optimization-AI/LibAUC



Training and evaluation scripts to reproduce both 2D and 3D experiments are available in PyTorch


# Code Structure
* [`MedMNIST2D/`](./MedMNIST2D/): training and evaluation scripts of MedMNIST2D
  * [`models.py`](./MedMNIST2D/models.py): *ResNet-18* and *ResNet-50* models (for small-image datasets like CIFAR-10/100)
  * [`train_and_eval_pytorch.py`](./MedMNIST2D/train_and_eval_pytorch.py): training and evaluation script implemented with PyTorch

* [`MedMNIST3D/`](./MedMNIST3D/): training and evaluation scripts of MedMNIST3D
  * [`models.py`](./MedMNIST3D/models.py): *ResNet-18* and *ResNet-50* models (for small-image datasets like CIFAR-10/100), basically same as [`MedMNIST2D/models.py`](./MedMNIST2D/models.py)
  * [`train_and_eval_pytorch.py`](./MedMNIST3D/train_and_eval_pytorch.py): training and evaluation script implemented with PyTorch
    

# Installation and Requirements
This repository is working with [MedMNIST official code](https://github.com/MedMNIST/MedMNIST/) and PyTorch.

1. Setup the required environments and install `medmnist` as a standard Python package:

    ```bash
      pip install medmnist
      pip install acsconv
      pip install tensorboardX
      pip install libauc
    ```

2. Clone this repository:

        ```bash
        git clone https://github.com/nimRobotics/csce_ml_experiments
        ```

3. Example run

        ```bash
        python csce_ml_experiments/MedMNIST2D/train_and_eval_pytorch.py \
        --data_flag breastmnist \
        --output_root './output2' \
        --num_epochs 100 \
        --batch_size 32 \
        --download \
        --model_flag resnet18 \
        --optimizer adam \
        --rotation 15 \
        --scale \
        --translation
        ```

4. Outputs are provided with the submission.

5. All models: https://drive.google.com/drive/folders/1wKJIWhKXI5GV5gHLT6cHf47bo2I1hLnI?usp=share_link


# Usage 2D data

Here is the documentation for the command-line options used in this script:

## `--data_flag`

- **Description:** A string argument that specifies the dataset to be used. Default is `pathmnist`.
- **Type:** `str`

## `--output_root`

- **Description:** A string argument that specifies the root directory where the output files (models and results) will be saved. Default is `./output`.
- **Type:** `str`

## `--num_epochs`

- **Description:** An integer argument that specifies the number of training epochs. If set to 0, the script will only test the model. Default is `100`.
- **Type:** `int`

## `--gpu_ids`

- **Description:** A string argument that specifies the ID of the GPU to be used. Default is `0`.
- **Type:** `str`

## `--batch_size`

- **Description:** An integer argument that specifies the batch size for training. Default is `128`.
- **Type:** `int`

## `--download`

- **Description:** A boolean flag that specifies whether to download the dataset or not.
- **Type:** `boolean`

## `--resize`

- **Description:** A boolean flag that specifies whether to resize the images from size 28x28 to 224x224.
- **Type:** `boolean`

## `--as_rgb`

- **Description:** A boolean flag that specifies whether to convert the grayscale image to RGB.
- **Type:** `boolean`

## `--model_path`

- **Description:** A string argument that specifies the root directory of the pre-trained model to be tested.
- **Type:** `str`

## `--model_flag`

- **Description:** A string argument that specifies the backbone of the model to be used. The options are `resnet18` and `resnet50`. Default is `resnet18`.
- **Type:** `str`

## `--run`

- **Description:** A string argument that specifies the name of the standard evaluation CSV file to be saved. The file name is in the format `{flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv`. Default is `model1`.
- **Type:** `str`

## `--test_flag`

- **Description:** A boolean flag that specifies whether to test the model or not.
- **Type:** `boolean`

## `--libauc_loss`

- **Description:** A boolean flag that specifies whether to use the library AUC loss function or not.
- **Type:** `boolean`

## `--optimizer`

- **Description:** A string argument that specifies the optimizer to be used. The options are `adam`, `sgd`, and `pesg`. Default is `adam`.
- **Type:** `str`

## `--rotation`

- **Description:** An integer argument that specifies the angle of rotation for data augmentation. Default is `None`.
- **Type:** `int`

## `--scale`

- **Description:** A boolean flag that specifies whether to apply scaling for data augmentation or not.
- **Type:** `boolean`

## `--translation`

- **Description:** A boolean flag that specifies whether to apply translation for data augmentation or not.
- **Type:** `boolean`


# Usage 3D data


# Command-Line Options Documentation

Here is the documentation for the command-line options used in this script:

## `--data_flag`

- **Description:** A string argument that specifies the dataset to be used. Default is `organmnist3d`.
- **Type:** `str`

## `--output_root`

- **Description:** A string argument that specifies the root directory where the output files (models) will be saved. Default is `./output`.
- **Type:** `str`

## `--num_epochs`

- **Description:** An integer argument that specifies the number of training epochs. If set to 0, the script will only test the model. Default is `100`.
- **Type:** `int`

## `--gpu_ids`

- **Description:** A string argument that specifies the ID of the GPU to be used. Default is `0`.
- **Type:** `str`

## `--batch_size`

- **Description:** An integer argument that specifies the batch size for training. Default is `32`.
- **Type:** `int`

## `--conv`

- **Description:** A string argument that specifies the converter to be used. The options are `Conv2_5d`, `Conv3d`, and `ACSConv`. Default is `ACSConv`.
- **Type:** `str`

## `--pretrained_3d`

- **Description:** A string argument that specifies the type of pre-trained model to be used. Default is `i3d`.
- **Type:** `str`

## `--download`

- **Description:** A boolean flag that specifies whether to download the dataset or not.
- **Type:** `boolean`

## `--as_rgb`

- **Description:** A boolean flag that specifies whether to copy channels and transform the shape of the input from 1x28x28x28 to 3x28x28x28.
- **Type:** `boolean`

## `--shape_transform`

- **Description:** A boolean flag that specifies whether to multiply 0.5 at evaluation for shape dataset.
- **Type:** `boolean`

## `--model_path`

- **Description:** A string argument that specifies the root directory of the pre-trained model to be tested.
- **Type:** `str`

## `--model_flag`

- **Description:** A string argument that specifies the backbone of the model to be used. The options are `resnet18` and `resnet50`. Default is `resnet18`.
- **Type:** `str`

## `--run`

- **Description:** A string argument that specifies the name of the standard evaluation CSV file to be saved. The file name is in the format `{flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv`. Default is `model1`.
- **Type:** `str`

## `--test_flag`

- **Description:** A boolean flag that specifies whether to test the model or not.
- **Type:** `boolean`

## `--libauc_loss`

- **Description:** A boolean flag that specifies whether to use the library AUC loss function or not.
- **Type:** `boolean`

## `--optimizer`

- **Description:** A string argument that specifies the optimizer to be used. The options are `adam`, `sgd`, and `pesg`. Default is `adam`.
- **Type:** `str`

## `--rotation`

- **Description:** An integer argument that specifies the angle of rotation for data augmentation. Default is `None`.
- **Type:** `int`

## `--scale`

- **Description:** A boolean flag that specifies whether to apply scaling for data augmentation or not.
- **Type:** `boolean`

## `--translation`

- **Description:** A boolean flag that specifies whether to apply translation for data augmentation or not.
- **Type:** `boolean`
