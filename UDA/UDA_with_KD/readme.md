### Prerequisites:
- python == 3.6.8
- pytorch ==1.1.0
- torchvision == 0.3.0
- numpy, scipy, sklearn, PIL, argparse, tqdm


### Training:
##### Unsupervised Domain Adaptation (UDA) on the Digits dataset with Data-free Knowledge Distillation
- MNIST -> USPS (**m2u**)
	```python
	 python uda_digit.py --dset m2u_half --gpu_id 0 --output ckps_digits --cls_par 0.1 --issplit True, --mode student --bottleneck 42
	```
