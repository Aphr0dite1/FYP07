# Color Cast Removal
Python implementations of "[Color Cast Removal for Image Restoration in Low Light / Night Time Conditions with Model-Based Learning]"

### Already Implemented
- histogram equalization(he)
- dynamic histogram equalization(dhe)
- Our Color Cast Removal Model for Image Restoration in Low Light / Night Time Using Exposure Fusion Framework

### Requirements
- scipy
- numpy
- imageio
- matplotlib
- cv2
- skimage

### Usage
If you want the result of "Our Color Cast Removal Model for Image Restoration in Low Light / Night Time Using Exposure Fusion Framework"
```
python fusion.py <input image> 
```
If you want the result of "[A Dynamic Histogram Equalization for Image Contrast Enhancement](https://ieeexplore.ieee.org/document/4266947/)"
```
python dhe.py <input image>
```
If you want the result of histogram equalization
```
python he.py <input image>
```

### Results
<p align='center'>
  <img src='Color-Cast-Removal/low/1.png' height='256' width='192'/>
  <img src='Color-Cast-Removal/result/fusion_low_result/1.png' height='256' width='192'/>
  <img src='Color-Cast-Removal/low/780.png' height='256' width='192'/>
  <img src='Color-Cast-Removal/result/fusion_low_result/780.png' height='256' width='192'/>
</p>

<p align='center'>
  <img src='Color-Cast-Removal/low/708.png' height='252' width='384'/>
  <img src='Color-Cast-Removal/result/fusion_low_result/708.png' height='252' width='384'/>
</p>

<p align='center'>
  <img src='Color-Cast-Removal/low/669.png' height='252' width='384'/>
  <img src='Color-Cast-Removal/result/fusion_low_result/669.png' height='252' width='384'/>
</p>
