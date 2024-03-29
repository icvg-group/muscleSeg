# muscleSeg
In this repo we provide the inference pipeline of our muscle segmetation model. The inputs are any abdominal CT, at L3 level, with size 512×512, and the output is the predictions of the masks for the following muscle regions and the Vertebral body:
1. rectus abdominus muscle
2. trans abd,int and ext obl
3. psoas major mucle
4. quardratus lumborum muscle
5. eretor spinae muscle
6. L3 Vertebral body

## Prerequirement
This repo relies on the following python dependancy:
* pytorch>=1.8
* torchvision
* segmentation-models-pytorch==0.3.2
* pydicom
* numpy
* Pillow
* OpenCV2


## How to run the ccode
Simply execute the command line  script `python muscleSeg [device] [dicom file]`

The device can be cpu or cuda (for using GPU).
