## Running the sample

1.  Compile this TensorRT repository as described in main page  

2.  Copy data forder from repository root: 
	```
	cp -r data build/out
	```

3.  Run sample with this command:
	```
	cd build/out
	LD_LIBRARY_PATH=. ./sample_onnx_mnist_tanhshrink
	```

4.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_onnx_mnist_tanhshrink # ./sample_onnx_mnist_tanhshrink
	[04/13/2021-09:35:25] [I] Building and running a GPU inference engine for Onnx MNIST
	Create parser
	Construct Network
	[04/13/2021-09:36:01] [I] [TRT] ----------------------------------------------------------------
	[04/13/2021-09:36:01] [I] [TRT] Input filename:   data/mnist/mnist_cc_trt.onnx
	[04/13/2021-09:36:01] [I] [TRT] ONNX IR version:  0.0.7
	[04/13/2021-09:36:01] [I] [TRT] Opset version:    13
	[04/13/2021-09:36:01] [I] [TRT] Producer name:    
	[04/13/2021-09:36:01] [I] [TRT] Producer version: 
	[04/13/2021-09:36:01] [I] [TRT] Domain:           
	[04/13/2021-09:36:01] [I] [TRT] Model version:    0
	[04/13/2021-09:36:01] [I] [TRT] Doc string:       
	[04/13/2021-09:36:01] [I] [TRT] ----------------------------------------------------------------
	Build engine
	[04/13/2021-09:36:12] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
	[04/13/2021-09:36:15] [I] [TRT] Detected 1 inputs and 1 output network tensors.
	Build All
	[04/13/2021-09:36:15] [I] Input:
	[04/13/2021-09:36:15] [I] @@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@%++@@@@@@@@@
	@@@@@@@@@@@@@%=.   =@@@@@@@@
	@@@@@@@@@@@%-   ..  %@@@@@@@
	@@@@@@@@@@-    .%%  #@@@@@@@
	@@@@@@@@@*    .%@%  #@@@@@@@
	@@@@@@@@@:  .-%@@*  #@@@@@@@
	@@@@@@@@@%=#%@@@@+  #@@@@@@@
	@@@@@@@@@@@@@@@@@-  #@@@@@@@
	@@@@@@@@@@@@@++++: :@@@@@@@@
	@@@@@@@@@@@#:     .%@@@@@@@@
	@@@@@@@@@#.  .-    +@@@@@@@@
	@@@@@@@@*  +#@@.    *@@@@@@@
	@@@@@@@=  :@@@:  =-  :###@@@
	@@@@@@=  +@@#: .#@@#-   .@@@
	@@@@@+  *@#-  =%@@@@@*::%@@@
	@@@@@   ..   +@@@@@@@@@@@@@@
	@@@@@.    :+%@@@@@@@@@@@@@@@
	@@@@@@*=+%@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	
	[04/13/2021-09:36:15] [I] Output:
	[04/13/2021-09:36:15] [I]  Prob 0  0.0000 Class 0: 
	[04/13/2021-09:36:15] [I]  Prob 1  0.0000 Class 1: 
	[04/13/2021-09:36:15] [I]  Prob 2  1.0000 Class 2: **********
	[04/13/2021-09:36:15] [I]  Prob 3  0.0000 Class 3: 
	[04/13/2021-09:36:15] [I]  Prob 4  0.0000 Class 4: 
	[04/13/2021-09:36:15] [I]  Prob 5  0.0000 Class 5: 
	[04/13/2021-09:36:15] [I]  Prob 6  0.0000 Class 6: 
	[04/13/2021-09:36:15] [I]  Prob 7  0.0000 Class 7: 
	[04/13/2021-09:36:15] [I]  Prob 8  0.0000 Class 8: 
	[04/13/2021-09:36:15] [I]  Prob 9  0.0000 Class 9: 
	[04/13/2021-09:36:15] [I] 
	```

