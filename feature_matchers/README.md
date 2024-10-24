# Feature Matchers
This folder contains [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master), [LightGlue](https://github.com/cvg/LightGlue/tree/edb2b838efb2ecfe3f88097c5fad9887d95aedad), matchers with SIFT and ORB.
## Installation 
Clone the repo:
```
git clone https://github.com/rongirl/thermal-slam-dockers.git  
cd multi-view-stereo-dockers/feature_matchers
```
Install dependencies:
```
pip install --no-cache-dir -r requirements.txt
```
## Weights for SuperGlue
Weights for indoor images can be downloaded with the following command:
```
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_indoor.pth
```
For outdoor images, use this command:
```
wget https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_outdoor.pth
```
## Running SuperGlue
You should provide list of pairs ```--input_pairs``` for images contained in ```--input_dir```.
Images can be resized before network inference with ```--resize```, the default 
```--resize``` is 640x480.
```
python3 main.py --input_dir <IMAGES_PATH> \
--matcher superglue \
--path_to_weights <WEIGHTS_PATH> \
--resize <int int> 
--output_dir <OUTPUT_PATH> \
--input_pairs <IMAGES_PAIRS_PATH>
```
## Running LightGlue
```
python3 main.py --input_dir <IMAGES_PATH> \
--matcher lightglue \
--output_dir <OUTPUT_PATH> \
--input_pairs <IMAGES_PAIRS_PATH>
```
## Running matcher with SIFT
```
python3 main.py --input_dir <IMAGES_PATH> \
--matcher sift \
--output_dir <OUTPUT_PATH> \
--input_pairs <IMAGES_PAIRS_PATH>
```
## Running matcher with ORB
```
python3 main.py --input_dir <IMAGES_PATH> \
--matcher orb \
--output_dir <OUTPUT_PATH> \
--input_pairs <IMAGES_PAIRS_PATH>
```
