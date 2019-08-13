![GuidePic](https://github.com/VitoRazor/Lidar_RGB_detector/blob/master/images/3Ddetector.png)
# Lidar with RGB 
This project is used to verify the usefulness of RGB channels for 3D Object detection.
The project matches the RGB pixels from the image to the lidar point cloud through the camera model. When constructing data, add RGB channels to points. Then, load the data into the model for training or prediction. To some extent, 3D AP is improved.

The detector comes from the project, [second.pytorch](https://github.com/nutonomy/second.pytorch)
Thank you very much for the author's tireless efforts, from which I learned a lot.

### Performance in KITTI lidar validation set (50/50 split)


```car.fhd.config``` + 100 epochs + super converge (about 2 days ) +  (15 fps in gt 1060):

```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.53, 89.63, 88.12
bev  AP:90.03, 87.57, 86.47
3d   AP:88.37, 77.83, 76.15
```

```car.fhd.onestage.config``` + 50 epochs + super converge (6.5 hours) +  (25 fps in 1080Ti):

```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.65, 89.59, 88.72
bev  AP:90.38, 88.20, 86.98
3d   AP:89.16, 78.78, 77.41
```

### Performance in KITTI lidar with RGB validation set (50/50 split)


```car.fhd.config``` + 100 epochs + super converge (about 2 days ) +  (10 fps in GT1060):

```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.63, 89.83, 88.32
bev  AP:90.13, 87.57, 86.47
3d   AP:88.67, 78.33, 76.65
```

```car.fhd.onestage.config``` + 50 epochs + super converge (1 day) +  (10 fps in GT1060):

```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.65, 89.59, 88.83
bev  AP:90.53, 88.65, 87.31
3d   AP:89.35, 78.83, 78.16
```

## Prepare dataset of lidar with RGB

![GuidePic](https://github.com/VitoRazor/Lidar_RGB_detector/blob/master/images/lidar_RGB.JPG)

* KITTI Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   ├── velodyne_reduced <-- empty directory
       |   └── velodyne_rgb
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           ├── velodyne_reduced <-- empty directory
           └── velodyne_rgb
```

Then run
```bash
python create_data.py kitti_data_prep --data_path=KITTI_DATASET_ROOT
```
## train

```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir
```
## evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
```

