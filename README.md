# CSE 291G Project: Video Completion

## Supplementing Materials

https://drive.google.com/drive/folders/1FuMb18YNEhMmrsJ9fD37_aN5u57eF0mG?usp=sharing

The link above contains the pre-trained KTH first order motion checkpoint, and samples of generated videos (to remove suspicion of cherry-picking) using baseline models and our models. The file structure is as follows:

* `convolution` BAIR test videos predicted by CVI (convolutional video inbetweening)
* `convolution_kth` KTH test videos predicted by CVI 
* `GANgenerated[1,2,3]` BAIR test videos predicted by adversarial generative motion model with random noise vector [1, 2, or 3]
* `GANgenerated[1,2,3]_kth` BAIR test videos predicted by adversarial generative motion model with random noise vector [1, 2, or 3]
* `generated_kth` KTH test videos predicted by autoregressive generative motion model
* `generated` BAIR test videos predicted by autoregressive generative motion model
* `videos` BAIR ground truth videos
* `videos_kth` KTH ground truth videos



## Preprocessing

Download the KTH videos (in AVI format) from https://www.csc.kth.se/cvap/actions/.

Run `processing-kth.ipynb` in the folder `first-order-model` to 

* Center-crop the KTH videos and resize to (64, 64), (128, 128) or (256, 256).
* Save the videos in mp4 format
* Break down long videos to multiple 16-frames videos

Download the BAIR dataset (in tfrecords format) from http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar.

Run `first-order-model-bair.ipynb`  in the folder `first-order-model` to 

* Read the `tfrecord` videos, resize to (64, 64), (128, 128) or (256, 256), save to mp4 if you want



## First Order Motion Model

We train our [**First Order Motion Model for Image Animation**](<https://arxiv.org/abs/2003.00196>) based on Aliaksandr Siarohin's official implementation. We made a series of minor changes to the repository to make it support greyscale video dataset such as KTH and to output each epoch's dataloader progress, as each epoch can takes up to half an hour.

If you use the original implementation and set the dataset number of channels to 1, you may meet several incompatible shape problems during backprop, and may not be able to visualize the keypoints at the end of each epoch.

### Usage

First download the official repository,

```
git clone https://github.com/AliaksandrSiarohin/first-order-model.git
```

Replace the corresponding files in the repository by the files in the folder `first-order-model`. Place the preprocessed kth videos in mp4 format in a folder only containing those videos. Place the `kth-128.yaml` in the config file. Then run

```
CUDA_VISIBLE_DEVICES=0,1
setsid python run.py --config config/kth-128.yaml --device_ids 0,1 --checkpoint log/kth-128_if_your_training_stops_middle_way > mylog 2>&1 &
tail -f mylog
```

Note that the model would diverge due to bilinear update if your pytorch version is not 1.0.0. Recommend establish a venv and install all the dependencies using the `requirement.txt` in the official repository.

###  Estimate Train Time

Our model takes ~40 hours with two Nvidia Titan XP. The pretrained checkpoint is publicly available in the supplemental material folder.



## Video_Inbetweening

This implementation of [**From Here to There: Video Inbetweening Using Direct 3D Convolutions**](https://arxiv.org/pdf/1905.10240.pdf) is based on @[wangwangbuaa](https://github.com/wangwangbuaa)'s unofficial tensorflow implementation. We train two convolutional models for the project milestone but finally decide to switch to KTH dataset and use pretrained models at `https://tfhub.dev/google/tweening_conv3d_bair/1` and `https://tfhub.dev/google/tweening_conv3d_kth/1`. 

### Download dataset

```
./data/KTH/download.sh
```

### Training

```
python3 train_kth_multigpu.py --gpu 0 --batch_size 32 --lr 0.0001
```

To train other dataset, create a text list of training video file names, and then revise the following two lines in `train_kth_multigpu.py`

```
data_path = "../data/MITD/"
f = open(data_path + "eruption_train.txt", "r")
```

Load the videos with `load_data('KTH')` means last two items of each row of the text list are the `low` and `high` values. Load the videos with `load_data('MITD')` automatically sets the `low` and `high` values; the text list only needs to contain the file names.

### Testing

```
python test.py --p [checkpoint iteration] --gpu 0 --prefix [checkpoint prefix directory]
```

Example Usage

```
python test.py --p 9002 --gpu 0 --prefix KTH_MCNET_gpu_id=0_image_size=64_K=2_T=14_batch_size=32_alpha=1.0_beta=0.02_lr=0.0001_num_layer=15
```

### Use Pre-trained Model

Run `CVI.ipynb` in the parent folder to

* Get the first and last frames of the mp4 test videos 
* Load the frames with `batch-size=16 `, as the pretrained models require this batch size
* Load the pretrained models, fill the intermediate frames and save the complete videos as mp4



## Generative Motion Model

Download BAIR official checkpoint from [Google Drive](https://drive.google.com/drive/folders/1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH.).

Place `first-order-model-bair.ipynb` and `first-order-model-kth.ipynb` under the official repository, replace `demo.py` with the updated version to

* Extract keypoints and jacobians from train videos and format them
* Load generated keypoints and jacobians, send them to the dense motion module and the occulusion-aware module to yield the generated inbetweening videos
* Demo the generated videos as image sequences or play them in HTML players
* Perform linear interpolation

### Adversarial

Run `Project_WGAN` ipynb to 

* Train an adversarial generative motion model given the keypoints and jacobians from train videos
* Predict all keypoints and jacobians given the keypoints and jacobians from test videos
* Plot the D and G cost curves.



