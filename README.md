# Adaptive Data Augmentation For Aspect Sentiment Quad Prediction
<hr style="border: 1px solid gray">
This is the PyTorch implementation for the paper [Adaptive Data Augmentation For Aspect Sentiment Quad Prediction](https://arxiv.org/abs/2401.06394), which is accepted by ICASSP 2024.
> Wenyuan Zhang, Xinghua Zhang, Shiyao Cui, Kun Huang, Xuebin Wang, Tingwen Liu. Adaptive Data Augmentation For Aspect Sentiment Quad Prediction. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing 2024 (ICASSP 24'), 14-19 April, 2024, COEX, Seoul, Corea.
### Requirments
* python 3.7.6
* pytorch 1.11.0 + cuda 11.3
* pytorch-lightning 1.3.5
### Run step
#### step1
To set the dataset path and the save path for the augmented dataset, you can modify the corresponding variables in the code.
After that, you can execute the retrieval-augmented.py file to generate the augmented dataset.
#### step2
To set the augmented dataset path, execute run.sh file.

If you find this work helpful to your research, please kindly consider citing our paper.
```
@article{Zhang2024AdaptiveDA,
  title={Adaptive Data Augmentation for Aspect Sentiment Quad Prediction},
  author={Wenyuan Zhang and Xinghua Zhang and Shiyao Cui and Kun Huang and Xuebin Wang and Tingwen Liu},
  year={2024},
  journal={arXiv preprint arXiv:2401.06394},
}
```
