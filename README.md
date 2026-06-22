# Radio3DMix

Radio3DMix is a large-scale synthetic 3D radio map dataset and generation toolkit for urban low-altitude wireless scenarios. It is used in the RadioGen3D framework for data-efficient 3D radio map estimation.

## Dataset Overview

| Item | Value |
| --- | --- |
| Total radio maps | 50,000 |
| Building regions | 200 Beijing urban regions |
| Radio map size | 256 x 256 x 20 voxels |
| Spatial resolution | 1 m |
| Transmitters per sample | 2 |
| Coefficient combinations | 25 |
| Realizations per building map | 10 |

The repository provides the scripts and source environment data needed to generate Radio3DMix. The generated full radio-map array is not committed because the default output file is very large.

## Repository Contents

```text
.
|-- Fitting_Test_Fast.py        # Fast Radio3DMix radio-map generator
|-- RadioUNet_Model.py          # Wrapper for the pretrained 2D RadioUNet model
|-- modules.py                  # Network modules used by RadioUNet
|-- Visualization.py            # Visualization helper for generated .npy files
|-- shp2png.py                  # Shapefile-to-PNG preprocessing utility
|-- models/
|   |-- Trained_Model_FirstU.pt
|   `-- Trained_Model_SecondU.pt
`-- data/
    `-- Beijing_Dataset/
        |-- 256_square/         # Original Beijing shapefile tiles
        `-- BJ_256_png/         # Rasterized building maps
```

The following local files are intentionally not included: `BJ_256_png_dilation/`, `Fitting_Test.py`, `Two_Map_Plot_Merge_TwoWay_Resize.py`, IDE caches, Python caches, and generated `results/` files.

## Environment

This repository uses Git LFS for the pretrained `.pt` model weights. After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch installation may depend on your CUDA version. If the command above does not install a suitable PyTorch build, install `torch` and `torchvision` from the official PyTorch instructions first.

## Generate Radio3DMix

The default parameters in `Fitting_Test_Fast.py` match the dataset description above:

```bash
python Fitting_Test_Fast.py
```

This generates:

```text
results/signal_map_c25_m200_r10_tx2_h20.npy
results/tx_locations_c25_m200_r10_tx2_h20.csv
```

The full default `.npy` file requires more than 240 GB of disk space. For a small sanity check, run:

```bash
python Fitting_Test_Fast.py --maximum-orders 2 --num-realizations 1 --p2-list 2.0 --delta-list 0.0 --num-tx-list 2
```

Useful default parameters:

```text
--building-path data/Beijing_Dataset/BJ_256_png
--num-tx-list 2
--minimum-orders 1
--maximum-orders 200
--maximum-heights 20
--num-realizations 10
--p2-list 2.0,2.1,2.2,2.3,2.4
--delta-list 0.0,0.1,0.2,0.3,0.4
```

## Building Map Preprocessing

The repository already includes rasterized building maps in `data/Beijing_Dataset/BJ_256_png/`. To regenerate them from the shapefiles:

```bash
python shp2png.py --shp-path data/Beijing_Dataset/256_square --output-dir data/Beijing_Dataset/BJ_256_png --max-height 50
```

The Radio3DMix generator uses the first 20 height slices by default.

## Visualization

After generating a radio-map `.npy` file and its transmitter-location `.csv`, use:

```bash
python Visualization.py
```

The default visualization settings are kept small and can be edited in the script according to the generated file name.
