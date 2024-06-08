# Fast HumanNeRF

## Prerequisite
Python==3.8.0

CUDA==11.7

### Create environment
To get started, you need to create and activate a virtual environment. This ensures that all dependencies are managed and do not conflict with other projects.
```
1. Create and activate the virtual environment:
$ conda create --name fasthumannerf python==3.8
$ conda activate fasthumannerf

2. Install PyTorch and other necessary libraries:
$ pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

3. Install additional Python packages:
$ pip install -r requirements.txt
$ pip install hydra-core --upgrade

4. Ensure the correct CUDA version is set:
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.7/bin:$PATH
```


### Download SMPL model
#### 1. Download the SMPL model:
- Go to [SMPL](https://smplify.is.tue.mpg.de/) and download the gender-neutral SMPL model.

- Unpack the downloaded file **mpips_smplify_public_v2.zip**.

#### 2. Copy the SMPL model to the appropriate directory:

- Copy the smpl model.
```
SMPL_DIR=/path/to/smpl

MODEL_DIR=$SMPL_DIR/smplify_public/code/models

cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models
```


#### 3. Remove Chumpy objects from the SMPL model:
- Follow the instructions provided [here](https://github.com/vchoutas/smplx/tree/master/tools) to modify the SMPL model appropriately.


## Run on ZJU-Mocap Dataset

### Prepare a dataset

#### 1. Download the ZJU-Mocap dataset:
- Follow the instructions [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) to download the ZJU-Mocap dataset.

#### 2. Modify the configuration file for subject 377:
- Open `tools/prepare_zju_mocap/377.yaml` and modify the `zju_mocap_path` to point to the directory where the ZJU-Mocap dataset is located:

```
dataset:
    zju_mocap_path: /path/to/zju_mocap
    subject: '377'
    sex: 'neutral'

...
```

#### 3. Run the data preprocessing script:
```
python ./tools/prepare_zju_mocap/prepare_dataset.py
```

- Repeat the same process for other subjects by modifying their respective configuration files and running the preprocessing script.

## Train
#### Select the subject configuration in train.py:
- Open train.py and ensure the correct subject configuration is set.
```
python train.py
```
    

## Evaluate
#### Select the subject configuration in eval.py and run
```
python eval.py
```
   

## Render output
Select the subject configuration in run.py and run

    python run.py

