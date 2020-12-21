# Predicting the Evolution of Photospheric Magnetic Field in Solar Active Regions Using Deep Learning

Liang Bai, Yi Bi, Bo Yang, Jun-Chao Hong, Zhe Xu, Zhen-Hong Shang, Hui Liu, Hai-Sheng Ji and [Kai-Fan Ji<sup>*</sup>](https://github.com/jikaifan)
> The continuous observation of the magnetic field by Solar DynamicsObservatory (SDO)/Helioseismic and Magnetic Imager (HMI) produces numerous image sequences in time and space. These sequences provide data support for predicting the evolution of photospheric magnetic field. Based on the spatiotemporal long short-termmemory(LSTM) network, we use the preprocessed data of photospheric magnetic field in active regions to build a prediction model for magnetic field evolution. Detailed experimental results and discussions show that the proposed model could effectively predict the large-scale and short-term evolution of the photospheric magnetic field in active regions. Moreover, our study may provide a reference for the spatiotemporal prediction of other solar activities. 


## Environments
- Ubuntu 18.04
- Pytorch 1.5+
- CUDA 10.0 & cuDNN 7.1
- Python 3.6

## Datasets and Model
**Paper:** https://arxiv.org/abs/2012.03584

**Datasets:** [baiduYunPan](https://pan.baidu.com/s/1tJ9IdVF4GqTD0oAqtp8fxA) code: 6i6b

**Trained Model:** [baiduYunPan](https://pan.baidu.com/s/1Vwn9wDj6Jns6R_qxRV-XBg) code: sgjw

**More Test Results:** [animation](animation/)
## Run
- train the model
```bash
#download the datasets, and put them in to  ../datasets/MF_datasets
python run.py --is_training=True --device=cuda:0
```
- generate the test results
```bash
#download the trained model, and put it to the out_path
python run.py --is_training=False --device=cuda:0 --pretrained_model=out_path/model.ckpt
```

## Acknowledgement
Our work and implementations are inspired by and based on
MIM (<https://github.com/Yunbo426/MIM>) and MIM_Pytorch (<https://github.com/coolsunxu/MIM_Pytorch>) 
