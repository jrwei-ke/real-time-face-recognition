# Real-time_Facial_Detect_Insightface-paddle

This repo refers to [Insight_Face_Paddle](https://github.com/littletomatodonkey/insight-face-paddle) and [Real-time Multi-Camera Facial Recognition](https://github.com/M-M-Akash/Face_Recognition_System). These two repos have a few lib incompatible problems and that is why I bulid this repo to deliver a proper way to build up.

## Enviroment

It is not recommended to use M-series processor for MacOS as base enviroment for the reason that it may report lots of incompatible problems in spite of the compatible announcement from officail website. Win10 and Ubuntu 22.04 are recommended and this repo is operated on ubuntu 22.04 desktop system. 

### Using conda to build virtual environment:

```linux
conda create --name env_name python==3.8
```
[Insight_Face_Paddle](https://github.com/littletomatodonkey/insight-face-paddle) list a range of compatible version of python, however, version 3.8 is strongly suggested.  

### Activate virtual environment:

```linux
conda activate env_name
```

### Install paddlepaddle lib:

```python
pip install paddlepaddle==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Install [Insight_Face_Paddle](https://github.com/littletomatodonkey/insight-face-paddle) through wheel:

```
pip install wheel
git clone https://github.com/littletomatodonkey/insight-face-paddle.git
cd insight-face-paddle
Python setup.py bdist_wheel
pip install dist/* 
```

This repo has another way to install by pip. I tried and found lots of incompatible issues for the reason that lots of libs auto installed by pip are getting higher version.  

### Install paddlehub:

```python
pip install paddlehub==2.1.0
```

This hub is for accessing to pre-trained facial-detection model.

### Version modification for libs:

```python
pip install protobuf==3.20 
Pip install paddlenlp==2.5.2
```

Incompatible result from the auto updating of pip, downgrade version to avoid this problem.

### Install pre-trained facial-detection model

```python
hub install pyramidbox_lite_mobile
```


