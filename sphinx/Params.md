## Datasets

```yaml
input_shape:
  celeba:
    image:
      - 54
      - 44
      - 3
    label:
      - 2
  cifar10:
    image:
      - 32
      - 32
      - 3
    label:
      - 10
  cifar100:
    image:
      - 32
      - 32
      - 3
    label:
      - 100
  femnist:
    image:
      - 28
      - 28
      - 1
    label:
      - 62
  mnist:
    image:
      - 28
      - 28
      - 1
    label:
      - 10
```



## Machine Leaning Models

```yaml
# MLP
activation: relu
dropout: 0.2
lr: 0.0005
optimizer: sgd
units:
- 512
- 512
# LeNet
activation: relu
lr: 0.0005
optimizer: sgd
pooling: max
```

## Federated Models

```

```

