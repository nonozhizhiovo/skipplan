## 2022/09/24

# CrossTask
(i) **Set-up Dataset**. We provide two ways to step-up the dataset for CrossTask [1]. You can **either** use pre-extracted features
```
cd datasets/CrossTask_assets
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://vision.eecs.yorku.ca/WebShare/CrossTask_s3d.zip
unzip '*.zip'
```
**or** extract features from raw video using the following code (* Both options work, pick one to use) 
```
cd raw_data_process
python download_CrossTask_videos.py
python InstVids2TFRecord_CrossTask.py
bash lmdb_encode_CrossTask.sh 1 1
```

## Baseline train

```
python train_cont.py
```

# 2022/10/07

## bidirectional
```
python train_bi.py
```
## bidirectionnal + hierarchical

```
python train_bihi.py
```

