# Exporting RNN model to onnx and caffe with speed comparison in pyTorch-1

The full tutorial is taken from pyTorch's official documentation. (https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#). I have added code to export the trained model (in tutorial) to first onnx and then running it using caffe2.

*Results*:

1. Time taken for torch model:
```
predict('Dovesky')
predict('Jackson')
predict('Satoshi')

Output:
> Dovesky
--- 0.0013060569763183594 seconds ---
(-0.71) Russian
(-0.98) Czech
(-2.98) English

> Jackson
--- 0.0005233287811279297 seconds ---
(-0.77) Scottish
(-1.81) English
(-2.04) Russian

> Satoshi
--- 0.0004475116729736328 seconds ---
(-0.96) Japanese
(-1.07) Arabic
(-2.47) Italian
```

2. Time taken onnx model:
```
predictCaffe2Model('Dovesky')
predictCaffe2Model('Jackson')
predictCaffe2Model('Satoshi')

Output:

> Dovesky
--- 0.03229641914367676 seconds ---
(-0.50) Russian
(-1.12) Czech
(-3.48) Polish

> Jackson
--- 0.022897005081176758 seconds ---
(-0.22) Scottish
(-2.65) English
(-2.95) Russian

> Satoshi
--- 0.022278785705566406 seconds ---
(-0.74) Arabic
(-1.42) Japanese
(-2.41) Italian
```
