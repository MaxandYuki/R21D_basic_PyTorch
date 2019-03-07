# R21D_basic_PyTorch
*still in development...*

This is the repository of my R(2+1)D model based on ResNet-18. Development architecture is PyTorch 1.0.

Details about R(2+1)D can be referred from *[A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/pdf/1711.11248 "A Closer Look at Spatiotemporal Convolutions for Action Recognition")* (Du Tran .etc)

## Prepare dateset

```bash
cd data/kinetics400
mkdir access && cd access
ln -s $YOUR_KINETICS400_DATASET_TRAIN_DIR$ RGB_train
ln -s $YOUR_KINETICS400_DATASET_VAL_DIR$ RGB_val
```
## Train & test
Executing shell script in the scripts directory:

Train:
`./scripts/train_kinetics400_21d.sh`

Test:
`./scripts/test_kinetics400_21d.sh`
## Performance
Training and testing environment: 8 GTX 1080 Ti GPUs

Input sample size: 16 * 3 * 16 * 112 * 112 on each GPU

**Results:**

|  Clip@1 | Clip@5  |  Video@1 |  Video@5 |
| ------------ | ------------ | ------------ | ------------ |
| 60.750  | 83.076  |  67.588 | 88.190  |

| Clip avg  |  Video avg |
| ------------ | ------------ |
|  71.913 | 77.889  |


------------


Thanks to Dr. Wang's guiding and Lei Zhou's help.