![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.10](https://img.shields.io/badge/PyTorch->=1.10-blue.svg)

# [NeruIPS2024] RLE: A Unified Perspective of Data Augmentation for Cross-Spectral Re-identification
The official repository for RLE: A Unified Perspective of Data Augmentation for Cross-Spectral Re-identification [[pdf]](https://arxiv.org/pdf/2411.01225)

### Prepare Datasets

- (1) RegDB Dataset [3]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 
  
- (2) SYSU-MM01 Dataset [4]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to prepare the dataset, the training data will be stored in ".npy" format.

## Training

We utilize 1 3090 GPU for training and you can train the model with:

```bash
python train.py --gpu 'your device id' --dataset 'sysu or regdb'
```

1. RLE is a data augmentation strategy. You can change the model as needed.

**Some examples:**
```bash
# SYSU-MM01
python train.py --gpu 0 --dataset sysu
```

## Evaluation
```bash
python test.py --mode 'mode for SYSU-MM01' --resume 'model_path' --gpu 'your device id' --dataset 'sysu or regdb'
```

**Some examples:**
```bash
# SYSU-MM01
python test.py --mode all --resume sysu.t --gpu 0 --dataset sysu
```


## Citation
Please kindly cite this paper in your publications if it helps your research:
```bash
@article{tan2024rle,
  title={RLE: A Unified Perspective of Data Augmentation for Cross-Spectral Re-Identification},
  author={Tan, Lei and Zhang, Yukang and Han, Keke and Dai, Pingyang and Zhang, Yan and WU, YONGJIAN and Ji, Rongrong},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={126977--126996},
  year={2024}
}
```
## Acknowledgement
Our code is based on [Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [1] and [DEEN](https://github.com/ZYK100/LLCM) [2]

## References
[1] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C., Hoi. Deep learning for person re-identification: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

[2] Y. Zhang, and H. Wang. Diverse embedding expansion network and low-light cross-modality benchmark for visible-infrared person re-identification. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

## Contact

If you have any question, please feel free to contact us. E-mail: [tanlei@stu.xmu.edu.cn](mailto:tanlei@stu.xmu.edu.cn)
