# Skip-Plan: Procedure Planning in Instructional Videos via Condensed Action Space Learning
Zhiheng Li, Geng Wenjia, Muheng Li, Lei Chen, Yansong Tang, Jiwen Lu, Jie Zhou
## Install Dependency
* `conda create --channel conda-forge --name procedureFormer python=3.7.3`
* `conda activate procedureFormer`
* `conda install --file requirements.txt`

This code assumes CUDA support.

## Train and test on CrossTask dataset: 
(i) Train

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

(ii) Test on the pretrained model: 

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

If you find this code useful in your work then please cite


## Contact
Please contact Zhiheng Li @ lizhihan21@mails.tsinghua.edu.cn if any issue.

## References

