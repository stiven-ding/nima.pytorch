# PyTorch NIMA v1
Originally published by truskovskiyk (https://github.com/truskovskiyk/nima.pytorch). 

Modifications will save your time to run this on Google Colab / Kaggle Code. Run with Python 3.6.

PyTorch implementation of [Neural IMage Assessment](https://arxiv.org/abs/1709.05424) by Hossein Talebi and Peyman Milanfar. You can learn more from [this post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html). 


## Installing on Google Colab / Kaggle

Check out [Google Colab setup](https://colab.research.google.com/gist/stiven-ding/b39d9673985030dc9acb668648b4a50b/nima-v1.ipynb?authuser=2).

```bash
# Install PIP
!curl https://bootstrap.pypa.io/get-pip.py -o get-pip3.6.py
!python3.6 get-pip3.6.py

# Install NIMA
!git clone https://github.com/stiven-ding/nima.pytorch.git
%cd nima.pytorch
!python3.6 -m pip install -r requirements.txt
!python3.6 setup.py install
```

## Pretrained model  

To test with the pretrained model, run
```bash
!curl -O https://s3-us-west-1.amazonaws.com/models-nima/pretrain-model.pth 
!python3.6 nima/cli.py get-image-score --path_to_model_weight ./pretrain-model.pth --path_to_image test_image.jpg
```

You can use this [pretrain-model](https://s3-us-west-1.amazonaws.com/models-nima/pretrain-model.pth) with
```bash
val_emd_loss = 0.079
test_emd_loss = 0.080
```

## Dataset

The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf)
You can get it from [here](https://github.com/mtobeiyf/ava_downloader)
Here are some examples of images with theire scores 
![result1](https://3.bp.blogspot.com/-_BuiLfAsHGE/WjgoftooRiI/AAAAAAAACR0/mB3tOfinfgA5Z7moldaLIGn92ounSOb8ACLcBGAs/s1600/image2.png)

## Model 

Used MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

## Usage
```bash
export PYTHONPATH=.
export PATH_TO_AVA_TXT=/storage/DATA/ava/AVA.txt
export PATH_TO_IMAGES=/storage/DATA/images/
export PATH_TO_CSV=/storage/DATA/ava/
export BATCH_SIZE=16
export NUM_WORKERS=2
export NUM_EPOCH=50
export INIT_LR=0.0001
export EXPERIMENT_DIR_NAME=/storage/experiment_n0001
```
Clean and prepare dataset
```bash
python3.6 nima/cli.py prepare-dataset --path_to_ava_txt $PATH_TO_AVA_TXT \
                                    --path_to_save_csv $PATH_TO_CSV \
                                    --path_to_images $PATH_TO_IMAGES

```

Train model
```bash
python3.6 nima/cli.py train-model --path_to_save_csv $PATH_TO_CSV \
                                --path_to_images $PATH_TO_IMAGES \
                                --batch_size $BATCH_SIZE \
                                --num_workers $NUM_WORKERS \
                                --num_epoch $NUM_EPOCH \
                                --init_lr $INIT_LR \
                                --experiment_dir_name $EXPERIMENT_DIR_NAME


```
Use tensorboard to tracking training progress

```bash
tensorboard --logdir .
```
Validate model on val and test datasets
```bash
python3.6 nima/cli.py validate-model --path_to_model_weight ./pretrain-model.pth \
                                    --path_to_save_csv $PATH_TO_CSV \
                                    --path_to_images $PATH_TO_IMAGES \
                                    --batch_size $BATCH_SIZE \
                                    --num_workers $NUM_EPOCH
```
Get scores for one image
```bash
python3.6 nima/cli.py get-image-score --path_to_model_weight ./pretrain-model.pth --path_to_image test_image.jpg
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [neural-image-assessment in keras](https://github.com/titu1994/neural-image-assessment)
* [Neural-IMage-Assessment in pytorch](https://github.com/kentsyx/Neural-IMage-Assessment)
* [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)
* [origin NIMA article](https://arxiv.org/abs/1709.05424)
* [origin MobileNetV2 article](https://arxiv.org/pdf/1801.04381)
* [Post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html)
* [Heroku: Cloud Application Platform](https://www.heroku.com/)
