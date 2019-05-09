# AgnosticMC

## Environment:
- Python 3.5.3
- PyTorch 0.4.1

## Dataset
CIFAR10 and MNIST datasets are required here. When you first run the script below, 
if you haven't downloaded the data before, it will automatically download them at folder `data_CIFAR10` and `data_MNIST`. 
Stay patient. 

## Script:
First, please go to the `Bin_CIFAR10` folder. For now, we only update the codes in that case.
There are a lot of loss terms, described in the `argparse` part, if the loss weight (shorted as `lw` in the code) is 0, 
it means this loss term will not be included in the total loss.

test DFL using random noise as input on MNIST:
```shell
# debug mode, print log on screen
python main.py  --lw_soft 100  --lw_hard_dec 0  --lw_hard_se 0  --use_random_input  --gpu <id>  

# formal experiment mode, print log in backend
nohup python main.py  --lw_soft 100  --lw_hard_dec 0  --lw_hard_se 0  --use_random_input  --gpu <id> --CodeID <code git log id>  > /dev/null &
```
The log will be saved in folder `../Experiment/xxx`, where `xxx` is a folder named by the time stamp when you run 
the code. For details, please refer to the `set_up_dir` function in `util.py`.  An example for `xxx`: `SERVER218-20190501-124737_test`. 
It tells you where and when you run the experiment, easy for later check.
