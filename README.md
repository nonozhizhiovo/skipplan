# Skip-Plan: Procedure Planning in Instructional Videos via Condensed Action Space Learning
[Zhiheng Li](https://nonozhizhiovo.github.io/), Wenjia Geng, Muheng Li, Lei Chen, [Yansong Tang](https://andytang15.github.io/), Jiwen Lu, Jie Zhou
## Installation
python==3.8.17

Install other packages `pip install -r requirements.txt`

This code assumes CUDA support.

## Download and Set-up CrossTask Dataset
```
cd datasets/CrossTask_assets
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://vision.eecs.yorku.ca/WebShare/CrossTask_s3d.zip
unzip '*.zip'
```

## Download pretrained models
Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1_8dwpin7IAagE3f9e01TTpaz3uqpcn7E?usp=sharing).
Arrange pretrained models into the path `checkpoint/CrossTask_t3 or 4 or 5 or 6_best.pth.tar`


## Train and test on CrossTask dataset: 
### (i) Train

T = 3: 
```
python train_cont.py
```

T = 4: 
```
python train_tower4.py
```
T = 5: 
```
python train_tower5.py
```

T = 6: 
```
python train_tower6.py
```

### (ii) Test the pretrained model: 

T = 3: 
```
python test_cont.py
```

T = 4: 
```
python test_tower4.py
```
T = 5: 
```
python test_tower5.py
```
T = 6: 
```
python test_tower6.py
```

## Citation

If you find this code useful in your work then please cite:

```
@inproceedings{li2023skip,
  title={Skip-Plan: Procedure Planning in Instructional Videos via Condensed Action Space Learning},
  author={Li, Zhiheng and Geng, Wenjia and Li, Muheng and Chen, Lei and Tang, Yansong and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10297--10306},
  year={2023}
}
```

## Contact
Please contact Zhiheng Li @ lizhihan21@mails.tsinghua.edu.cn if any issue.

## Acknowledgements

This code is built on [P3IV](https://github.com/SamsungLabs/procedure-planning). We thank the authors for sharing their codes and extracted features.

