# FOR.ai Technical Challenge 

### Download Instructions
```
git clone https://github.com/arakhmat/for.ai.challenge
```

### Usage
If you running the script for the first time:
```
# Specify '--train' flag to train the model
python challenge.py -t
```
For consequent runs, use:
```
python challenge.py
```
You can specify the following arguments when running the script:
```
'-t', type=bool,  action='store_true', help='train the network'
'-e', type=bool,  action='store_true', help='run evaluation'
'-s', type=int,   default=20000,       help='number of training steps'
'-b', type=int,   default=64,          help='training batch size'
'-o', type=int,   default=2,           help='label to replace'
'-n', type=int,   default=6,           help='adversarial label'
'-r', type=int,   default=10,          help='number of images to replace'
'-l', type=float, default=0.001,       help='adversarial learning rate'
 ```
    
## Acknowledgments
* [A Guide to TF Layers: Building a Convolutional Neural Network](https://www.tensorflow.org/get_started/mnist/pros#deep-mnist-for-experts) by 
* [Breaking Linear Classifiers on ImageNet](http://karpathy.github.io/2015/03/30/breaking-convnets/) by Andrej Karpathy
* [Explaining and Harnessing Adversarial Examples]](https://arxiv.org/pdf/1412.6572.pdf) by Ian J. Goodfellow, Jonathon Shlens and Christian Szegedy
