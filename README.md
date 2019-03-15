# Exporting RNN model to onnx and caffe with speed comparison in pyTorch-1

The full tutorial is taken from pyTorch's official documentation. (https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#). I have added code to export the trained model (in tutorial) to first onnx and then running it using caffe2.

Results:

```
1. Time taken for torch model:

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

```
2. Time taken onnx model:
predictCaffe2Model('Dovesky')
predictCaffe2Model('Jackson')
predictCaffe2Model('Satoshi')

Output:
> Dovesky
--- 0.023533344268798828 seconds ---
(-1.69) English
(-1.98) Irish
(-2.11) Scottish

> Jackson
--- 0.02212238311767578 seconds ---
(-1.54) Irish
(-2.11) English
(-2.14) Dutch

> Satoshi
--- 0.022337913513183594 seconds ---
(-0.92) Italian
(-1.96) Japanese
(-2.58) Korean
```
