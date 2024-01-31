# DSPNet

**Abstract**:

As dominant Few-shot Semantic Segmentation (FSS) methods, the prototypical scheme suffers from a fundamental limitation: The pooling-based prototypes are prone to losing local details. The complicated and diverse details in medical
images amplify this problem considerably. Unlike conventional incremental solution that constructs new prototypes to capture more details, this paper introduces a novel Detail Self-refined Prototype Network (DSPNet). Our core idea is 
enhancing theprototypes’ ability to model details via detail self-refining. To this end, we propose two new attention-like designs. In foreground semantic prototype attention module, to construct global semantics while maintaining the
captured detail semantics, we fuse cluster-based detail prototypes as a single class prototype in a channel-wise weighting fashion. In background channel-structural multi-head attention module, considering that the complicated background
often has no apparent semantic relation in the spatial dimensions,we integrate each background detail prototype’s channel structural information for its self-enhancement. Specifically, we introduce a neighbour channel-aware regulation
into the multi-head channel attention, exploiting a local-global adjustment mechanism.Elements of each detail prototype are individually refreshed by different heads in BCMA. Extensive experiments on two challenging medical benchmarks
demonstrate the superiority of DSPNet over previous state-of-the-art FSS methods.
**NOTE: We are actively updating this repository**

If you find this code base useful, please cite our paper. Thanks!

```
@article{Song Tang2024DSPNet,
  title={Few-Shot Medical Image Segmentation with Detail Self-Refined Prototypes},
  author={Song Tang, Shaxu Yan, Xiaozhi Qi, Jianxin Gao, Mao Ye , Jianwei Zhang and Xiatian Zhu},
  journal={},
  year={2024}
}
```

### 1. Dependencies

Please install essential dependencies (see `requirements.txt`) 

```
dcm2nii
nibabel==2.5.1
numpy==1.21.6
opencv-python==4.1.1
Pillow==9.5.0 
sacred==0.7.5
scikit-image==0.14.0
SimpleITK==1.2.3
torch==1.8.1
torchvision==0.9.1
```

### 2. Data pre-processing 

### Datasets and pre-processing
Download:  
**Abdominal MRI**

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 fold) to `nii` files in 3D for the ease of reading

run `./data/CHAOST2/dcm_img_to_nii.sh` to convert dicom images to nifti files.

run `./data/CHAOST2/png_gth_to_nii.ipynp` to convert ground truth with `png` format to nifti.

2. Pre-processing downloaded images

run `./data/CHAOST2/image_normalize.ipynb`

**Abdominal CT**

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABS/` directory

1. Intensity windowing 

run `./data/SABS/intensity_normalization.ipynb` to apply abdominal window.

2. Crop irrelavent emptry background and resample images

run `./data/SABS/resampling_and_roi.ipynb` 

**Shared steps**

3. Build class-slice indexing for setting up experiments

run `./data/<CHAOST2/SABS>class_slice_index_gen.ipynb`  

### Training  
1. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints `./pretrained_model/hub/checkpoints` folder,
2. Run `training.py` 

#### Note:  
The α and (w1, w2, w3) coefficient for  in the code `./models/alpmodule.py` should be manually modified.  
For setting 1, ABD:  α = 0.3, (w1, w2, w3) = (0.2, 0.8, 0.2)   CMR：α = 0.3, (w1, w2, w3) = (0.1, 0.9, 0.1)
For setting 2, ABD： α = 0.2, (w1, w2, w3) = (0.3, 0.6, 0.3) 

### Testing
Run `validation.py`

### Acknowledgement
This code is based on [SSL-ALPNet](https://arxiv.org/abs/2007.09886v2) (ECCV'20) by [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git)
