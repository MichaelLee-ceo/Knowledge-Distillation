# Knowledge-Distillation
Implement knowledge distillation for knowledge transferring.


### Dataset
- CIFAR10
- FashionMNIST


### Baseline
CIFAR10
|  | 3-layer CNN (Student) | Resnet18 (Teacher) |
|:-:|:-:|:-:|
| Accuracy | 83.46% | 93.27% |
| + Mixup  | 84.54% | 94.52% |

FashionMNIST
|  | 3-layer CNN (Student) | Resnet18 (Teacher) |
|:-:|:-:|:-:|
| Accuracy | 92.43% | 94.66% |
| + Mixup  | 92.61% | 95.25% |

### Knowledge Distillation
*T: temperature, R: alpha rate*

CIFAR10
|  | Base| T=4, R=0.9 | T=4, R=0.95 | T=4, R=1.0 | T=10, R=0.9 |
|--|:-:|:-:|:-:|:-:|:-:|
| Accuracy | 83.46% | 84.47% | 84.86% | 84.14% | - |
| + Mixup  | 85.10% | -      | -      | -      | - |

FashionMNIST
|  | Base | T=4, R=0.9 | T=4, R=0.95 | T=4, R=1.0 | T=10, R=0.9 |
|--|:-:|:-:|:-:|:-:|:-:|
| Accuracy | 92.43% | 92.94% | 92.97% | 92.93% | 92.63% |