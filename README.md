# Predicting the Evolution of Photospheric Magnetic Field in Solar Active Regions Using Deep Learning

Liang Bai, Yi Bi, Bo Yang, Jun-Chao Hong, Zhe Xu, Zhen-Hong Shang, Hui Liu, Hai-Sheng Ji and [Kai-Fan Ji<sup>*</sup>](https://github.com/jikaifan)


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
## run
- train the model
```bash
python run.py --is_training=True --device=cuda:0
```
- generate the results
```bash
python run.py --is_training=False --device=cuda:0 --pretrained_model=outpath/checkpoints/model.ckpt-1
```

## Acknowledgement
Our work and implementations are inspired by and based on
MIM (<https://github.com/Yunbo426/MIM>) and MIM_Pytorch (<https://github.com/coolsunxu/MIM_Pytorch>) 
