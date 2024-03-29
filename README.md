# Exporting RNN model to onnx and caffe with speed comparison in pyTorch-1

The full tutorial is taken from pyTorch's official documentation. (https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#). I have added code to export the trained model (in tutorial) to first onnx and then running it using caffe2. Sample of code from notebook :

```
torch_out = torch.onnx.export(rnn, (lineToTensor('akansha')[0], rnn.initHidden()), 'char_rnn.onnx', export_params=True, verbose=True) # produces the RuntimeError below

Output : 

graph(%0 : Float(1, 57)
      %1 : Float(1, 128)
      %2 : Float(128, 185)
      %3 : Float(128)
      %4 : Float(18, 185)
      %5 : Float(18)) {
  %6 : Float(1, 185) = onnx::Concat[axis=1](%0, %1), scope: RNN
  %7 : Float(1, 128) = onnx::Gemm[alpha=1, beta=1, transB=1](%6, %2, %3), scope: RNN/Linear[i2h]
  %8 : Float(1, 18) = onnx::Gemm[alpha=1, beta=1, transB=1](%6, %4, %5), scope: RNN/Linear[i2o]
  %9 : Float(1, 18) = onnx::LogSoftmax[axis=1](%8), scope: RNN/LogSoftmax[softmax]
  return (%9, %7);
}
```

```
import onnx
import caffe2

# Load the ONNX ModelProto object. onnx_model is a standard Python protobuf object
onnx_model = onnx.load("char_rnn.onnx")

# takes array input instead of torch tensor
in1 = lineToTensor('akansha')[0].data.numpy()
in2 = rnn.initHidden().data.numpy()

start_time = time.time()
out, h = caffe2.python.onnx.backend.run_model(onnx_model, [in1, in2])
print("--- %s seconds ---" % (time.time() - start_time))

Output:

--- 0.008445978164672852 seconds ---
```
```
print(out)  # produces array output instead of torch tensor

Output:

array([[-3.0969102, -3.9964178, -4.4682226, -4.081335 , -1.6238663,
        -2.3298087, -3.6378086, -2.0644147, -4.3406234, -2.9473362,
        -2.8546565, -4.23935  , -4.4275675, -1.8626163, -2.8326497,
        -3.6823251, -4.080239 , -2.904221 ]], dtype=float32)
```
*Results of execution-time comparison*:

1. Time taken for torch model:
```
predict('Dovesky')
predict('Jackson')
predict('Satoshi')

Output:

> Dovesky
--- 0.004353046417236328 seconds ---
(-0.82) Czech
(-0.87) Russian
(-2.61) Polish

> Jackson
--- 0.00099945068359375 seconds ---
(-0.30) Scottish
(-2.00) English
(-3.30) Russian

> Satoshi
--- 0.0009684562683105469 seconds ---
(-1.03) Italian
(-1.75) Japanese
(-2.13) Polish
```

2. Time taken for onnx model using caffe2:
```
predictCaffe2Model('Dovesky')
predictCaffe2Model('Jackson')
predictCaffe2Model('Satoshi')

Output:

> Dovesky
--- 0.03211021423339844 seconds ---
(-0.82) Czech
(-0.87) Russian
(-2.61) Polish

> Jackson
--- 0.022547245025634766 seconds ---
(-0.30) Scottish
(-2.00) English
(-3.30) Russian

> Satoshi
--- 0.02180027961730957 seconds ---
(-1.03) Italian
(-1.75) Japanese
(-2.13) Polish


```
