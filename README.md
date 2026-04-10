# ST-RoomNet: Advanced Layout & Perspective Estimation

This project serves as a design and development framework for **Layout Estimation** and **Perspective Estimation**. It is a specialized development project derived and expanded from the original implementation.

Currently, it focuses on:
- **Layout Estimation**: Redefining room geometry from single images.
- **Perspective Estimation**: Refining spatial orientation and projective transformations.

This is based on the official implementation of ST-RoomNet: [Paper Link](https://openaccess.thecvf.com/content/CVPR2023W/VOCVALC/html/Ibrahem_ST-RoomNet_Learning_Room_Layout_Estimation_From_Single_Image_Through_Unsupervised_CVPRW_2023_paper.html)

The spatial transformer module is based on: [dantkz/spatial-transformer-tensorflow](https://github.com/dantkz/spatial-transformer-tensorflow)

## Key Enhancements
- Migrated for **TensorFlow 2.x** compatibility.
- Added advanced interpolation features (Nearest Neighbor, Bilinear, and Bicubic).
- Integrated **Git LFS** for handling large model weights efficiently.

## Prerequisites & Requirements
- **Git LFS**: Required to download model weights (`.h5` files). Run `git lfs pull` after cloning.
- **Python Packages**:
  - `opencv-python` (4.4.1+)
  - `tensorflow` (2.9.1+)
  - `tensorflow-addons`
  - `numpy`, `matplotlib`, `scipy`, `scikit-learn`

## Citation
If you use this code in your research, please cite the original paper:

```bibtex
@InProceedings{Ibrahem_2023_CVPR,
    author    = {Ibrahem, Hatem and Salem, Ahmed and Kang, Hyun-Soo},
    title     = {ST-RoomNet: Learning Room Layout Estimation From Single Image Through Unsupervised Spatial Transformations},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3375-3383}
}
```
