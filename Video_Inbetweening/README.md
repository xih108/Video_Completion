# Video_Inbetweening
Unofficially Tensorflow implementation of [**From Here to There: Video Inbetweening Using Direct 3D Convolutions**](<https://github.com/Fangyh09/Video_Inbetweening>)  (https://arxiv.org/pdf/1905.10240.pdf)


## update
Still training...

## Requirement
Tensorflow

## Model description 
T = 16 and D = 128 

input: (x_s, x_e), Gaussian noise vector u ∈ R^D

output:(x_s, xˆ1, . . . , xˆT −2, x_e)

## 0. Download Dataset KTH
```
./data/KTH/download.sh
```

## 1. Train
```
python3 train_kth_multigpu.py --gpu 0 --batch_size 32 --lr 0.0001
```

## Thanks
Some codes are from [here](https://github.com/BoPang1996/Deep-RNN-Framework).
