# Video_Inbetweening
This implementation of [**From Here to There: Video Inbetweening Using Direct 3D Convolutions**](<https://github.com/Fangyh09/Video_Inbetweening>)  (https://arxiv.org/pdf/1905.10240.pdf) is based on @[wangwangbuaa](https://github.com/wangwangbuaa)'s unofficial tensorflow implementation.

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