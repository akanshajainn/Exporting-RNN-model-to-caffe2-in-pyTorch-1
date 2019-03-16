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
--- 0.0007638931274414062 seconds ---
(-0.62) Russian
(-1.13) Czech
(-2.79) English

> Jackson
--- 0.0007193088531494141 seconds ---
(-0.36) English
(-2.08) Russian
(-2.46) Scottish

> Satoshi
--- 0.0006451606750488281 seconds ---
(-1.30) Italian
(-1.55) Japanese
(-1.74) Arabic
```

2. Time taken onnx model:
```
predictCaffe2Model('Dovesky')
predictCaffe2Model('Jackson')
predictCaffe2Model('Satoshi')

Output:

> Dovesky
--- 0.052504539489746094 seconds ---
(-0.62) Russian
(-1.13) Czech
(-2.79) English

> Jackson
--- 0.02758955955505371 seconds ---
(-0.36) English
(-2.08) Russian
(-2.46) Scottish

> Satoshi
--- 0.022540569305419922 seconds ---
(-1.30) Italian
(-1.55) Japanese
(-1.74) Arabic

```
