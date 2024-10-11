# **Weakly-Supervised 3D Hand Reconstruction with Knowledge Prior and Uncertainty Guidance** <br />
  [Yufei Zhang](https://zhangy76.github.io/), Jeffrey O. Kephart, Qiang Ji <br /> 
  ECCV2024, [arXiv](https://arxiv.org/abs/2407.12307) <br />
![](method.png)


This repository provides code for the proposed method and losses in "Weakly-Supervised 3D Hand Reconstruction with Knowledge Prior and Uncertainty Guidance" (also referred as KNOWN-Hand). 


## Environment Setup
```bash
conda create -n known python=3.7
conda activate known
pip install -r requirements.txt
```


## Model and Data Download
Please download the required data and trained model [assets](https://www.dropbox.com/scl/fo/i1hcyituupidft9ythj0m/AF5DFK8NurBFNBkFC4oMswg?rlkey=ag7htv1a2eycwpkqva5r9vpqr&dl=0) and directly overwrite the ./assets folder in the current directory. 


## Demo
Please run the following command to reconstruct 3D hands from videos using KNOWN-Hand. You may need an additional detector to extract hand regions if the hand is far from the camera.
```bash
python demo_video.py --video_path 'path to a testing video'
```


## Knowledge Prior
You can find the pose loss derived from hand biomechanics and functional anatomy knowledge in ```losses.py```.


## Citation
If you find our work useful, please consider citing the paper:
```bibtex
@article{zhang2024weakly,
  title={Weakly-Supervised 3D Hand Reconstruction with Knowledge Prior and Uncertainty Guidance},
  author={Zhang, Yufei and Kephart, Jeffrey O and Ji, Qiang},
  journal={arXiv preprint arXiv:2407.12307},
  year={2024}
}
```

If you have questions or encouter any issues when running the code, feel free to open an issue or directly contact me via: zhangy76@rpi.edu.

## References
The MANO model data is downloaded from [MANO model](https://mano.is.tue.mpg.de/). We thank them for generously sharing their outstanding work.

