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
Additionally, you can specify other arguments:
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
    
