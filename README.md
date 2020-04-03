# Fast-Neural-Style-Transfer
```sh
Implementation of Perceptual Losses for Real-Time Style Transfer and Super-Resolution.
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.
```

# Introduction
#### in Chinese
https://mp.weixin.qq.com/s/Ed-1fWOIhI52G-Ugrv7n9Q

# Environment
```sh
OS: Ubuntu 16.04
Graphics card: Titan xp
Python: python3.x with the packages in requirements.txt
```

# Usage
#### Train
```
Modify the cfg.py according to your need, and then run:
python train.py
```
#### Test
```
usage: Fast Neural Style [-h] --datapath DATAPATH --checkpointpath
                         CHECKPOINTPATH --outputpath OUTPUTPATH

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH   Path of image/video.
  --checkpointpath CHECKPOINTPATH
                        Path of checkpoint model.
  --outputpath OUTPUTPATH
                        Path to save results.
```

# Results
#### Picture
![img](./material/output.jpg)
#### Video
![giphy](./material/output.gif)

# References
```
[1]. https://github.com/jcjohnson/fast-neural-style
[2]. Perceptual Losses for Real-Time Style Transfer and Super-Resolution(Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li)
```

# More
#### WeChat Official Accounts
*Charles_pikachu*  
![img](./material/pikachu.jpg)