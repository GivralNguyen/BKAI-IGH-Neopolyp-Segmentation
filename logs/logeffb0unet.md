(tf2.3) quannm@quannm:/media/HDD/bkai$ python main.py
2021-11-06 17:02:07.607199: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Amount of images: 1000
Training: 800 - Validation: 200
100 - 26
2021-11-06 17:02:08.789748: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2021-11-06 17:02:08.834812: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:08.835242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 17:02:08.835300: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 17:02:08.837882: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 17:02:08.839943: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 17:02:08.840382: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 17:02:08.842552: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 17:02:08.843605: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 17:02:08.847483: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 17:02:08.847612: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:08.847808: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:08.847916: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 17:02:08.848196: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-06 17:02:08.854172: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3199980000 Hz
2021-11-06 17:02:08.854568: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56515c1d42a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-11-06 17:02:08.854586: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-11-06 17:02:08.904272: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:08.904464: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56515e096f50 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-11-06 17:02:08.904480: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce GTX 1070 Ti, Compute Capability 6.1
2021-11-06 17:02:08.904640: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:08.904750: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 17:02:08.904770: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 17:02:08.904794: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 17:02:08.904806: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 17:02:08.904817: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 17:02:08.904827: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 17:02:08.904837: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 17:02:08.904849: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 17:02:08.904894: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:08.905011: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:08.905095: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 17:02:08.905114: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 17:02:09.178806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-06 17:02:09.178834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2021-11-06 17:02:09.178840: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2021-11-06 17:02:09.179000: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:09.179202: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:02:09.179401: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6257 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Using unet_eff3 model 
Model: "MobilenetV2_Unet"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input (InputLayer)              [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
rescaling (Rescaling)           (None, 256, 256, 3)  0           input[0][0]                      
__________________________________________________________________________________________________
normalization (Normalization)   (None, 256, 256, 3)  7           rescaling[0][0]                  
__________________________________________________________________________________________________
stem_conv_pad (ZeroPadding2D)   (None, 257, 257, 3)  0           normalization[0][0]              
__________________________________________________________________________________________________
stem_conv (Conv2D)              (None, 128, 128, 32) 864         stem_conv_pad[0][0]              
__________________________________________________________________________________________________
stem_bn (BatchNormalization)    (None, 128, 128, 32) 128         stem_conv[0][0]                  
__________________________________________________________________________________________________
stem_activation (Activation)    (None, 128, 128, 32) 0           stem_bn[0][0]                    
__________________________________________________________________________________________________
block1a_dwconv (DepthwiseConv2D (None, 128, 128, 32) 288         stem_activation[0][0]            
__________________________________________________________________________________________________
block1a_bn (BatchNormalization) (None, 128, 128, 32) 128         block1a_dwconv[0][0]             
__________________________________________________________________________________________________
block1a_activation (Activation) (None, 128, 128, 32) 0           block1a_bn[0][0]                 
__________________________________________________________________________________________________
block1a_se_squeeze (GlobalAvera (None, 32)           0           block1a_activation[0][0]         
__________________________________________________________________________________________________
block1a_se_reshape (Reshape)    (None, 1, 1, 32)     0           block1a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block1a_se_reduce (Conv2D)      (None, 1, 1, 8)      264         block1a_se_reshape[0][0]         
__________________________________________________________________________________________________
block1a_se_expand (Conv2D)      (None, 1, 1, 32)     288         block1a_se_reduce[0][0]          
__________________________________________________________________________________________________
block1a_se_excite (Multiply)    (None, 128, 128, 32) 0           block1a_activation[0][0]         
                                                                 block1a_se_expand[0][0]          
__________________________________________________________________________________________________
block1a_project_conv (Conv2D)   (None, 128, 128, 16) 512         block1a_se_excite[0][0]          
__________________________________________________________________________________________________
block1a_project_bn (BatchNormal (None, 128, 128, 16) 64          block1a_project_conv[0][0]       
__________________________________________________________________________________________________
block2a_expand_conv (Conv2D)    (None, 128, 128, 96) 1536        block1a_project_bn[0][0]         
__________________________________________________________________________________________________
block2a_expand_bn (BatchNormali (None, 128, 128, 96) 384         block2a_expand_conv[0][0]        
__________________________________________________________________________________________________
block2a_expand_activation (Acti (None, 128, 128, 96) 0           block2a_expand_bn[0][0]          
__________________________________________________________________________________________________
block2a_dwconv_pad (ZeroPadding (None, 129, 129, 96) 0           block2a_expand_activation[0][0]  
__________________________________________________________________________________________________
block2a_dwconv (DepthwiseConv2D (None, 64, 64, 96)   864         block2a_dwconv_pad[0][0]         
__________________________________________________________________________________________________
block2a_bn (BatchNormalization) (None, 64, 64, 96)   384         block2a_dwconv[0][0]             
__________________________________________________________________________________________________
block2a_activation (Activation) (None, 64, 64, 96)   0           block2a_bn[0][0]                 
__________________________________________________________________________________________________
block2a_se_squeeze (GlobalAvera (None, 96)           0           block2a_activation[0][0]         
__________________________________________________________________________________________________
block2a_se_reshape (Reshape)    (None, 1, 1, 96)     0           block2a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block2a_se_reduce (Conv2D)      (None, 1, 1, 4)      388         block2a_se_reshape[0][0]         
__________________________________________________________________________________________________
block2a_se_expand (Conv2D)      (None, 1, 1, 96)     480         block2a_se_reduce[0][0]          
__________________________________________________________________________________________________
block2a_se_excite (Multiply)    (None, 64, 64, 96)   0           block2a_activation[0][0]         
                                                                 block2a_se_expand[0][0]          
__________________________________________________________________________________________________
block2a_project_conv (Conv2D)   (None, 64, 64, 24)   2304        block2a_se_excite[0][0]          
__________________________________________________________________________________________________
block2a_project_bn (BatchNormal (None, 64, 64, 24)   96          block2a_project_conv[0][0]       
__________________________________________________________________________________________________
block2b_expand_conv (Conv2D)    (None, 64, 64, 144)  3456        block2a_project_bn[0][0]         
__________________________________________________________________________________________________
block2b_expand_bn (BatchNormali (None, 64, 64, 144)  576         block2b_expand_conv[0][0]        
__________________________________________________________________________________________________
block2b_expand_activation (Acti (None, 64, 64, 144)  0           block2b_expand_bn[0][0]          
__________________________________________________________________________________________________
block2b_dwconv (DepthwiseConv2D (None, 64, 64, 144)  1296        block2b_expand_activation[0][0]  
__________________________________________________________________________________________________
block2b_bn (BatchNormalization) (None, 64, 64, 144)  576         block2b_dwconv[0][0]             
__________________________________________________________________________________________________
block2b_activation (Activation) (None, 64, 64, 144)  0           block2b_bn[0][0]                 
__________________________________________________________________________________________________
block2b_se_squeeze (GlobalAvera (None, 144)          0           block2b_activation[0][0]         
__________________________________________________________________________________________________
block2b_se_reshape (Reshape)    (None, 1, 1, 144)    0           block2b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block2b_se_reduce (Conv2D)      (None, 1, 1, 6)      870         block2b_se_reshape[0][0]         
__________________________________________________________________________________________________
block2b_se_expand (Conv2D)      (None, 1, 1, 144)    1008        block2b_se_reduce[0][0]          
__________________________________________________________________________________________________
block2b_se_excite (Multiply)    (None, 64, 64, 144)  0           block2b_activation[0][0]         
                                                                 block2b_se_expand[0][0]          
__________________________________________________________________________________________________
block2b_project_conv (Conv2D)   (None, 64, 64, 24)   3456        block2b_se_excite[0][0]          
__________________________________________________________________________________________________
block2b_project_bn (BatchNormal (None, 64, 64, 24)   96          block2b_project_conv[0][0]       
__________________________________________________________________________________________________
block2b_drop (Dropout)          (None, 64, 64, 24)   0           block2b_project_bn[0][0]         
__________________________________________________________________________________________________
block2b_add (Add)               (None, 64, 64, 24)   0           block2b_drop[0][0]               
                                                                 block2a_project_bn[0][0]         
__________________________________________________________________________________________________
block3a_expand_conv (Conv2D)    (None, 64, 64, 144)  3456        block2b_add[0][0]                
__________________________________________________________________________________________________
block3a_expand_bn (BatchNormali (None, 64, 64, 144)  576         block3a_expand_conv[0][0]        
__________________________________________________________________________________________________
block3a_expand_activation (Acti (None, 64, 64, 144)  0           block3a_expand_bn[0][0]          
__________________________________________________________________________________________________
block3a_dwconv_pad (ZeroPadding (None, 67, 67, 144)  0           block3a_expand_activation[0][0]  
__________________________________________________________________________________________________
block3a_dwconv (DepthwiseConv2D (None, 32, 32, 144)  3600        block3a_dwconv_pad[0][0]         
__________________________________________________________________________________________________
block3a_bn (BatchNormalization) (None, 32, 32, 144)  576         block3a_dwconv[0][0]             
__________________________________________________________________________________________________
block3a_activation (Activation) (None, 32, 32, 144)  0           block3a_bn[0][0]                 
__________________________________________________________________________________________________
block3a_se_squeeze (GlobalAvera (None, 144)          0           block3a_activation[0][0]         
__________________________________________________________________________________________________
block3a_se_reshape (Reshape)    (None, 1, 1, 144)    0           block3a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block3a_se_reduce (Conv2D)      (None, 1, 1, 6)      870         block3a_se_reshape[0][0]         
__________________________________________________________________________________________________
block3a_se_expand (Conv2D)      (None, 1, 1, 144)    1008        block3a_se_reduce[0][0]          
__________________________________________________________________________________________________
block3a_se_excite (Multiply)    (None, 32, 32, 144)  0           block3a_activation[0][0]         
                                                                 block3a_se_expand[0][0]          
__________________________________________________________________________________________________
block3a_project_conv (Conv2D)   (None, 32, 32, 40)   5760        block3a_se_excite[0][0]          
__________________________________________________________________________________________________
block3a_project_bn (BatchNormal (None, 32, 32, 40)   160         block3a_project_conv[0][0]       
__________________________________________________________________________________________________
block3b_expand_conv (Conv2D)    (None, 32, 32, 240)  9600        block3a_project_bn[0][0]         
__________________________________________________________________________________________________
block3b_expand_bn (BatchNormali (None, 32, 32, 240)  960         block3b_expand_conv[0][0]        
__________________________________________________________________________________________________
block3b_expand_activation (Acti (None, 32, 32, 240)  0           block3b_expand_bn[0][0]          
__________________________________________________________________________________________________
block3b_dwconv (DepthwiseConv2D (None, 32, 32, 240)  6000        block3b_expand_activation[0][0]  
__________________________________________________________________________________________________
block3b_bn (BatchNormalization) (None, 32, 32, 240)  960         block3b_dwconv[0][0]             
__________________________________________________________________________________________________
block3b_activation (Activation) (None, 32, 32, 240)  0           block3b_bn[0][0]                 
__________________________________________________________________________________________________
block3b_se_squeeze (GlobalAvera (None, 240)          0           block3b_activation[0][0]         
__________________________________________________________________________________________________
block3b_se_reshape (Reshape)    (None, 1, 1, 240)    0           block3b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block3b_se_reduce (Conv2D)      (None, 1, 1, 10)     2410        block3b_se_reshape[0][0]         
__________________________________________________________________________________________________
block3b_se_expand (Conv2D)      (None, 1, 1, 240)    2640        block3b_se_reduce[0][0]          
__________________________________________________________________________________________________
block3b_se_excite (Multiply)    (None, 32, 32, 240)  0           block3b_activation[0][0]         
                                                                 block3b_se_expand[0][0]          
__________________________________________________________________________________________________
block3b_project_conv (Conv2D)   (None, 32, 32, 40)   9600        block3b_se_excite[0][0]          
__________________________________________________________________________________________________
block3b_project_bn (BatchNormal (None, 32, 32, 40)   160         block3b_project_conv[0][0]       
__________________________________________________________________________________________________
block3b_drop (Dropout)          (None, 32, 32, 40)   0           block3b_project_bn[0][0]         
__________________________________________________________________________________________________
block3b_add (Add)               (None, 32, 32, 40)   0           block3b_drop[0][0]               
                                                                 block3a_project_bn[0][0]         
__________________________________________________________________________________________________
block4a_expand_conv (Conv2D)    (None, 32, 32, 240)  9600        block3b_add[0][0]                
__________________________________________________________________________________________________
block4a_expand_bn (BatchNormali (None, 32, 32, 240)  960         block4a_expand_conv[0][0]        
__________________________________________________________________________________________________
block4a_expand_activation (Acti (None, 32, 32, 240)  0           block4a_expand_bn[0][0]          
__________________________________________________________________________________________________
block4a_dwconv_pad (ZeroPadding (None, 33, 33, 240)  0           block4a_expand_activation[0][0]  
__________________________________________________________________________________________________
block4a_dwconv (DepthwiseConv2D (None, 16, 16, 240)  2160        block4a_dwconv_pad[0][0]         
__________________________________________________________________________________________________
block4a_bn (BatchNormalization) (None, 16, 16, 240)  960         block4a_dwconv[0][0]             
__________________________________________________________________________________________________
block4a_activation (Activation) (None, 16, 16, 240)  0           block4a_bn[0][0]                 
__________________________________________________________________________________________________
block4a_se_squeeze (GlobalAvera (None, 240)          0           block4a_activation[0][0]         
__________________________________________________________________________________________________
block4a_se_reshape (Reshape)    (None, 1, 1, 240)    0           block4a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4a_se_reduce (Conv2D)      (None, 1, 1, 10)     2410        block4a_se_reshape[0][0]         
__________________________________________________________________________________________________
block4a_se_expand (Conv2D)      (None, 1, 1, 240)    2640        block4a_se_reduce[0][0]          
__________________________________________________________________________________________________
block4a_se_excite (Multiply)    (None, 16, 16, 240)  0           block4a_activation[0][0]         
                                                                 block4a_se_expand[0][0]          
__________________________________________________________________________________________________
block4a_project_conv (Conv2D)   (None, 16, 16, 80)   19200       block4a_se_excite[0][0]          
__________________________________________________________________________________________________
block4a_project_bn (BatchNormal (None, 16, 16, 80)   320         block4a_project_conv[0][0]       
__________________________________________________________________________________________________
block4b_expand_conv (Conv2D)    (None, 16, 16, 480)  38400       block4a_project_bn[0][0]         
__________________________________________________________________________________________________
block4b_expand_bn (BatchNormali (None, 16, 16, 480)  1920        block4b_expand_conv[0][0]        
__________________________________________________________________________________________________
block4b_expand_activation (Acti (None, 16, 16, 480)  0           block4b_expand_bn[0][0]          
__________________________________________________________________________________________________
block4b_dwconv (DepthwiseConv2D (None, 16, 16, 480)  4320        block4b_expand_activation[0][0]  
__________________________________________________________________________________________________
block4b_bn (BatchNormalization) (None, 16, 16, 480)  1920        block4b_dwconv[0][0]             
__________________________________________________________________________________________________
block4b_activation (Activation) (None, 16, 16, 480)  0           block4b_bn[0][0]                 
__________________________________________________________________________________________________
block4b_se_squeeze (GlobalAvera (None, 480)          0           block4b_activation[0][0]         
__________________________________________________________________________________________________
block4b_se_reshape (Reshape)    (None, 1, 1, 480)    0           block4b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4b_se_reduce (Conv2D)      (None, 1, 1, 20)     9620        block4b_se_reshape[0][0]         
__________________________________________________________________________________________________
block4b_se_expand (Conv2D)      (None, 1, 1, 480)    10080       block4b_se_reduce[0][0]          
__________________________________________________________________________________________________
block4b_se_excite (Multiply)    (None, 16, 16, 480)  0           block4b_activation[0][0]         
                                                                 block4b_se_expand[0][0]          
__________________________________________________________________________________________________
block4b_project_conv (Conv2D)   (None, 16, 16, 80)   38400       block4b_se_excite[0][0]          
__________________________________________________________________________________________________
block4b_project_bn (BatchNormal (None, 16, 16, 80)   320         block4b_project_conv[0][0]       
__________________________________________________________________________________________________
block4b_drop (Dropout)          (None, 16, 16, 80)   0           block4b_project_bn[0][0]         
__________________________________________________________________________________________________
block4b_add (Add)               (None, 16, 16, 80)   0           block4b_drop[0][0]               
                                                                 block4a_project_bn[0][0]         
__________________________________________________________________________________________________
block4c_expand_conv (Conv2D)    (None, 16, 16, 480)  38400       block4b_add[0][0]                
__________________________________________________________________________________________________
block4c_expand_bn (BatchNormali (None, 16, 16, 480)  1920        block4c_expand_conv[0][0]        
__________________________________________________________________________________________________
block4c_expand_activation (Acti (None, 16, 16, 480)  0           block4c_expand_bn[0][0]          
__________________________________________________________________________________________________
block4c_dwconv (DepthwiseConv2D (None, 16, 16, 480)  4320        block4c_expand_activation[0][0]  
__________________________________________________________________________________________________
block4c_bn (BatchNormalization) (None, 16, 16, 480)  1920        block4c_dwconv[0][0]             
__________________________________________________________________________________________________
block4c_activation (Activation) (None, 16, 16, 480)  0           block4c_bn[0][0]                 
__________________________________________________________________________________________________
block4c_se_squeeze (GlobalAvera (None, 480)          0           block4c_activation[0][0]         
__________________________________________________________________________________________________
block4c_se_reshape (Reshape)    (None, 1, 1, 480)    0           block4c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block4c_se_reduce (Conv2D)      (None, 1, 1, 20)     9620        block4c_se_reshape[0][0]         
__________________________________________________________________________________________________
block4c_se_expand (Conv2D)      (None, 1, 1, 480)    10080       block4c_se_reduce[0][0]          
__________________________________________________________________________________________________
block4c_se_excite (Multiply)    (None, 16, 16, 480)  0           block4c_activation[0][0]         
                                                                 block4c_se_expand[0][0]          
__________________________________________________________________________________________________
block4c_project_conv (Conv2D)   (None, 16, 16, 80)   38400       block4c_se_excite[0][0]          
__________________________________________________________________________________________________
block4c_project_bn (BatchNormal (None, 16, 16, 80)   320         block4c_project_conv[0][0]       
__________________________________________________________________________________________________
block4c_drop (Dropout)          (None, 16, 16, 80)   0           block4c_project_bn[0][0]         
__________________________________________________________________________________________________
block4c_add (Add)               (None, 16, 16, 80)   0           block4c_drop[0][0]               
                                                                 block4b_add[0][0]                
__________________________________________________________________________________________________
block5a_expand_conv (Conv2D)    (None, 16, 16, 480)  38400       block4c_add[0][0]                
__________________________________________________________________________________________________
block5a_expand_bn (BatchNormali (None, 16, 16, 480)  1920        block5a_expand_conv[0][0]        
__________________________________________________________________________________________________
block5a_expand_activation (Acti (None, 16, 16, 480)  0           block5a_expand_bn[0][0]          
__________________________________________________________________________________________________
block5a_dwconv (DepthwiseConv2D (None, 16, 16, 480)  12000       block5a_expand_activation[0][0]  
__________________________________________________________________________________________________
block5a_bn (BatchNormalization) (None, 16, 16, 480)  1920        block5a_dwconv[0][0]             
__________________________________________________________________________________________________
block5a_activation (Activation) (None, 16, 16, 480)  0           block5a_bn[0][0]                 
__________________________________________________________________________________________________
block5a_se_squeeze (GlobalAvera (None, 480)          0           block5a_activation[0][0]         
__________________________________________________________________________________________________
block5a_se_reshape (Reshape)    (None, 1, 1, 480)    0           block5a_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5a_se_reduce (Conv2D)      (None, 1, 1, 20)     9620        block5a_se_reshape[0][0]         
__________________________________________________________________________________________________
block5a_se_expand (Conv2D)      (None, 1, 1, 480)    10080       block5a_se_reduce[0][0]          
__________________________________________________________________________________________________
block5a_se_excite (Multiply)    (None, 16, 16, 480)  0           block5a_activation[0][0]         
                                                                 block5a_se_expand[0][0]          
__________________________________________________________________________________________________
block5a_project_conv (Conv2D)   (None, 16, 16, 112)  53760       block5a_se_excite[0][0]          
__________________________________________________________________________________________________
block5a_project_bn (BatchNormal (None, 16, 16, 112)  448         block5a_project_conv[0][0]       
__________________________________________________________________________________________________
block5b_expand_conv (Conv2D)    (None, 16, 16, 672)  75264       block5a_project_bn[0][0]         
__________________________________________________________________________________________________
block5b_expand_bn (BatchNormali (None, 16, 16, 672)  2688        block5b_expand_conv[0][0]        
__________________________________________________________________________________________________
block5b_expand_activation (Acti (None, 16, 16, 672)  0           block5b_expand_bn[0][0]          
__________________________________________________________________________________________________
block5b_dwconv (DepthwiseConv2D (None, 16, 16, 672)  16800       block5b_expand_activation[0][0]  
__________________________________________________________________________________________________
block5b_bn (BatchNormalization) (None, 16, 16, 672)  2688        block5b_dwconv[0][0]             
__________________________________________________________________________________________________
block5b_activation (Activation) (None, 16, 16, 672)  0           block5b_bn[0][0]                 
__________________________________________________________________________________________________
block5b_se_squeeze (GlobalAvera (None, 672)          0           block5b_activation[0][0]         
__________________________________________________________________________________________________
block5b_se_reshape (Reshape)    (None, 1, 1, 672)    0           block5b_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5b_se_reduce (Conv2D)      (None, 1, 1, 28)     18844       block5b_se_reshape[0][0]         
__________________________________________________________________________________________________
block5b_se_expand (Conv2D)      (None, 1, 1, 672)    19488       block5b_se_reduce[0][0]          
__________________________________________________________________________________________________
block5b_se_excite (Multiply)    (None, 16, 16, 672)  0           block5b_activation[0][0]         
                                                                 block5b_se_expand[0][0]          
__________________________________________________________________________________________________
block5b_project_conv (Conv2D)   (None, 16, 16, 112)  75264       block5b_se_excite[0][0]          
__________________________________________________________________________________________________
block5b_project_bn (BatchNormal (None, 16, 16, 112)  448         block5b_project_conv[0][0]       
__________________________________________________________________________________________________
block5b_drop (Dropout)          (None, 16, 16, 112)  0           block5b_project_bn[0][0]         
__________________________________________________________________________________________________
block5b_add (Add)               (None, 16, 16, 112)  0           block5b_drop[0][0]               
                                                                 block5a_project_bn[0][0]         
__________________________________________________________________________________________________
block5c_expand_conv (Conv2D)    (None, 16, 16, 672)  75264       block5b_add[0][0]                
__________________________________________________________________________________________________
block5c_expand_bn (BatchNormali (None, 16, 16, 672)  2688        block5c_expand_conv[0][0]        
__________________________________________________________________________________________________
block5c_expand_activation (Acti (None, 16, 16, 672)  0           block5c_expand_bn[0][0]          
__________________________________________________________________________________________________
block5c_dwconv (DepthwiseConv2D (None, 16, 16, 672)  16800       block5c_expand_activation[0][0]  
__________________________________________________________________________________________________
block5c_bn (BatchNormalization) (None, 16, 16, 672)  2688        block5c_dwconv[0][0]             
__________________________________________________________________________________________________
block5c_activation (Activation) (None, 16, 16, 672)  0           block5c_bn[0][0]                 
__________________________________________________________________________________________________
block5c_se_squeeze (GlobalAvera (None, 672)          0           block5c_activation[0][0]         
__________________________________________________________________________________________________
block5c_se_reshape (Reshape)    (None, 1, 1, 672)    0           block5c_se_squeeze[0][0]         
__________________________________________________________________________________________________
block5c_se_reduce (Conv2D)      (None, 1, 1, 28)     18844       block5c_se_reshape[0][0]         
__________________________________________________________________________________________________
block5c_se_expand (Conv2D)      (None, 1, 1, 672)    19488       block5c_se_reduce[0][0]          
__________________________________________________________________________________________________
block5c_se_excite (Multiply)    (None, 16, 16, 672)  0           block5c_activation[0][0]         
                                                                 block5c_se_expand[0][0]          
__________________________________________________________________________________________________
block5c_project_conv (Conv2D)   (None, 16, 16, 112)  75264       block5c_se_excite[0][0]          
__________________________________________________________________________________________________
block5c_project_bn (BatchNormal (None, 16, 16, 112)  448         block5c_project_conv[0][0]       
__________________________________________________________________________________________________
block5c_drop (Dropout)          (None, 16, 16, 112)  0           block5c_project_bn[0][0]         
__________________________________________________________________________________________________
block5c_add (Add)               (None, 16, 16, 112)  0           block5c_drop[0][0]               
                                                                 block5b_add[0][0]                
__________________________________________________________________________________________________
block6a_expand_conv (Conv2D)    (None, 16, 16, 672)  75264       block5c_add[0][0]                
__________________________________________________________________________________________________
block6a_expand_bn (BatchNormali (None, 16, 16, 672)  2688        block6a_expand_conv[0][0]        
__________________________________________________________________________________________________
block6a_expand_activation (Acti (None, 16, 16, 672)  0           block6a_expand_bn[0][0]          
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 32, 32, 512)  1376768     block6a_expand_activation[0][0]  
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 32, 32, 752)  0           conv2d_transpose[0][0]           
                                                                 block4a_expand_activation[0][0]  
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 32, 32, 512)  3465728     concatenate[0][0]                
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 32, 32, 512)  2048        conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 32, 32, 512)  0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 512)  2359808     activation[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 512)  2048        conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 512)  0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 256)  524544      activation_1[0][0]               
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 64, 64, 400)  0           conv2d_transpose_1[0][0]         
                                                                 block3a_expand_activation[0][0]  
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, 64, 256)  921856      concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 64, 64, 256)  1024        conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 64, 256)  0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 64, 256)  590080      activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 64, 64, 256)  1024        conv2d_3[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 64, 64, 256)  0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 128, 128, 128 131200      activation_3[0][0]               
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 128, 128, 224 0           conv2d_transpose_2[0][0]         
                                                                 block2a_expand_activation[0][0]  
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 128, 128, 128 258176      concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 128, 128, 128 512         conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 128, 128, 128 0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 128, 128, 128 147584      activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 128, 128, 128 512         conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 128, 128, 128 0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 256, 256, 64) 32832       activation_5[0][0]               
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256, 256, 67) 0           conv2d_transpose_3[0][0]         
                                                                 input[0][0]                      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 256, 256, 64) 38656       concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 256, 256, 64) 256         conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 256, 256, 64) 0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 256, 256, 64) 36928       activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 256, 256, 64) 256         conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 256, 256, 64) 0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 256, 256, 3)  195         activation_7[0][0]               
==================================================================================================
Total params: 10,837,962
Trainable params: 10,816,611
Non-trainable params: 21,351
__________________________________________________________________________________________________
Epoch 1/20
2021-11-06 17:02:15.061216: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 17:02:15.609661: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
100/100 [==============================] - ETA: 0s - loss: 0.4082 - precision: 0.9550 - recall: 0.8345 - acc: 0.9106 
Epoch 00001: saving model to unet_eff3.h5
100/100 [==============================] - 35s 347ms/step - loss: 0.4082 - precision: 0.9550 - recall: 0.8345 - acc: 0.9106 - val_loss: 1.0379 - val_precision: 0.6263 - val_recall: 0.3005 - val_acc: 0.5811
Epoch 2/20
100/100 [==============================] - ETA: 0s - loss: 0.1672 - precision: 0.9769 - recall: 0.9706 - acc: 0.9738 
Epoch 00002: saving model to unet_eff3.h5
100/100 [==============================] - 34s 339ms/step - loss: 0.1672 - precision: 0.9769 - recall: 0.9706 - acc: 0.9738 - val_loss: 0.6035 - val_precision: 0.8563 - val_recall: 0.6890 - val_acc: 0.7630
Epoch 3/20
100/100 [==============================] - ETA: 0s - loss: 0.1259 - precision: 0.9815 - recall: 0.9771 - acc: 0.9793 
Epoch 00003: saving model to unet_eff3.h5
100/100 [==============================] - 34s 339ms/step - loss: 0.1259 - precision: 0.9815 - recall: 0.9771 - acc: 0.9793 - val_loss: 0.5015 - val_precision: 0.8402 - val_recall: 0.8044 - val_acc: 0.8239
Epoch 4/20
100/100 [==============================] - ETA: 0s - loss: 0.1027 - precision: 0.9841 - recall: 0.9808 - acc: 0.9823 
Epoch 00004: saving model to unet_eff3.h5
100/100 [==============================] - 34s 340ms/step - loss: 0.1027 - precision: 0.9841 - recall: 0.9808 - acc: 0.9823 - val_loss: 0.2409 - val_precision: 0.9491 - val_recall: 0.9484 - val_acc: 0.9488
Epoch 5/20
100/100 [==============================] - ETA: 0s - loss: 0.0874 - precision: 0.9858 - recall: 0.9822 - acc: 0.9839 
Epoch 00005: saving model to unet_eff3.h5
100/100 [==============================] - 34s 339ms/step - loss: 0.0874 - precision: 0.9858 - recall: 0.9822 - acc: 0.9839 - val_loss: 0.2432 - val_precision: 0.9459 - val_recall: 0.9342 - val_acc: 0.9406
Epoch 6/20
100/100 [==============================] - ETA: 0s - loss: 0.0739 - precision: 0.9881 - recall: 0.9842 - acc: 0.9858 
Epoch 00006: saving model to unet_eff3.h5
100/100 [==============================] - 34s 340ms/step - loss: 0.0739 - precision: 0.9881 - recall: 0.9842 - acc: 0.9858 - val_loss: 0.1754 - val_precision: 0.9623 - val_recall: 0.9553 - val_acc: 0.9593
Epoch 7/20
100/100 [==============================] - ETA: 0s - loss: 0.0657 - precision: 0.9895 - recall: 0.9846 - acc: 0.9870 
Epoch 00007: saving model to unet_eff3.h5
100/100 [==============================] - 34s 341ms/step - loss: 0.0657 - precision: 0.9895 - recall: 0.9846 - acc: 0.9870 - val_loss: 0.2090 - val_precision: 0.9563 - val_recall: 0.9476 - val_acc: 0.9508
Epoch 8/20
100/100 [==============================] - ETA: 0s - loss: 0.0565 - precision: 0.9903 - recall: 0.9868 - acc: 0.9885 
Epoch 00008: saving model to unet_eff3.h5
100/100 [==============================] - 34s 341ms/step - loss: 0.0565 - precision: 0.9903 - recall: 0.9868 - acc: 0.9885 - val_loss: 0.1658 - val_precision: 0.9586 - val_recall: 0.9322 - val_acc: 0.9409
Epoch 9/20
100/100 [==============================] - ETA: 0s - loss: 0.0509 - precision: 0.9910 - recall: 0.9878 - acc: 0.9893 
Epoch 00009: saving model to unet_eff3.h5
100/100 [==============================] - 34s 339ms/step - loss: 0.0509 - precision: 0.9910 - recall: 0.9878 - acc: 0.9893 - val_loss: 0.3822 - val_precision: 0.9201 - val_recall: 0.8183 - val_acc: 0.8713
Epoch 10/20
100/100 [==============================] - ETA: 0s - loss: 0.0449 - precision: 0.9915 - recall: 0.9893 - acc: 0.9903 
Epoch 00010: saving model to unet_eff3.h5
100/100 [==============================] - 34s 344ms/step - loss: 0.0449 - precision: 0.9915 - recall: 0.9893 - acc: 0.9903 - val_loss: 0.1958 - val_precision: 0.9480 - val_recall: 0.9335 - val_acc: 0.9401
Epoch 11/20
100/100 [==============================] - ETA: 0s - loss: 0.0377 - precision: 0.9929 - recall: 0.9915 - acc: 0.9921 
Epoch 00011: saving model to unet_eff3.h5
100/100 [==============================] - 34s 342ms/step - loss: 0.0377 - precision: 0.9929 - recall: 0.9915 - acc: 0.9921 - val_loss: 0.1187 - val_precision: 0.9730 - val_recall: 0.9712 - val_acc: 0.9719
Epoch 12/20
100/100 [==============================] - ETA: 0s - loss: 0.0359 - precision: 0.9927 - recall: 0.9915 - acc: 0.9920 
Epoch 00012: saving model to unet_eff3.h5
100/100 [==============================] - 34s 341ms/step - loss: 0.0359 - precision: 0.9927 - recall: 0.9915 - acc: 0.9920 - val_loss: 0.1699 - val_precision: 0.9572 - val_recall: 0.9534 - val_acc: 0.9546
Epoch 13/20
100/100 [==============================] - ETA: 0s - loss: 0.0315 - precision: 0.9935 - recall: 0.9925 - acc: 0.9930 
Epoch 00013: saving model to unet_eff3.h5
100/100 [==============================] - 34s 344ms/step - loss: 0.0315 - precision: 0.9935 - recall: 0.9925 - acc: 0.9930 - val_loss: 0.1317 - val_precision: 0.9712 - val_recall: 0.9702 - val_acc: 0.9706
Epoch 14/20
100/100 [==============================] - ETA: 0s - loss: 0.0272 - precision: 0.9945 - recall: 0.9939 - acc: 0.9942 
Epoch 00014: saving model to unet_eff3.h5
100/100 [==============================] - 34s 345ms/step - loss: 0.0272 - precision: 0.9945 - recall: 0.9939 - acc: 0.9942 - val_loss: 0.2817 - val_precision: 0.9106 - val_recall: 0.9025 - val_acc: 0.9064
Epoch 15/20
100/100 [==============================] - ETA: 0s - loss: 0.0231 - precision: 0.9953 - recall: 0.9948 - acc: 0.9950 
Epoch 00015: saving model to unet_eff3.h5
100/100 [==============================] - 35s 349ms/step - loss: 0.0231 - precision: 0.9953 - recall: 0.9948 - acc: 0.9950 - val_loss: 0.1140 - val_precision: 0.9699 - val_recall: 0.9681 - val_acc: 0.9688
Epoch 16/20
100/100 [==============================] - ETA: 0s - loss: 0.0228 - precision: 0.9951 - recall: 0.9946 - acc: 0.9948 
Epoch 00016: saving model to unet_eff3.h5
100/100 [==============================] - 35s 351ms/step - loss: 0.0228 - precision: 0.9951 - recall: 0.9946 - acc: 0.9948 - val_loss: 0.1172 - val_precision: 0.9712 - val_recall: 0.9695 - val_acc: 0.9702
Epoch 17/20
100/100 [==============================] - ETA: 0s - loss: 0.0219 - precision: 0.9951 - recall: 0.9946 - acc: 0.9948 
Epoch 00017: saving model to unet_eff3.h5
100/100 [==============================] - 35s 347ms/step - loss: 0.0219 - precision: 0.9951 - recall: 0.9946 - acc: 0.9948 - val_loss: 0.1746 - val_precision: 0.9588 - val_recall: 0.9570 - val_acc: 0.9578
Epoch 18/20
100/100 [==============================] - ETA: 0s - loss: 0.0191 - precision: 0.9957 - recall: 0.9954 - acc: 0.9955 
Epoch 00018: saving model to unet_eff3.h5
100/100 [==============================] - 34s 345ms/step - loss: 0.0191 - precision: 0.9957 - recall: 0.9954 - acc: 0.9955 - val_loss: 0.6455 - val_precision: 0.8921 - val_recall: 0.8900 - val_acc: 0.8909
Epoch 19/20
100/100 [==============================] - ETA: 0s - loss: 0.0159 - precision: 0.9965 - recall: 0.9962 - acc: 0.9963 
Epoch 00019: saving model to unet_eff3.h5
100/100 [==============================] - 34s 343ms/step - loss: 0.0159 - precision: 0.9965 - recall: 0.9962 - acc: 0.9963 - val_loss: 0.2322 - val_precision: 0.9404 - val_recall: 0.9389 - val_acc: 0.9395
Epoch 20/20
100/100 [==============================] - ETA: 0s - loss: 0.0154 - precision: 0.9965 - recall: 0.9962 - acc: 0.9964 
Epoch 00020: saving model to unet_eff3.h5

Epoch 00020: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-06.
100/100 [==============================] - 34s 343ms/step - loss: 0.0154 - precision: 0.9965 - recall: 0.9962 - acc: 0.9964 - val_loss: 0.2335 - val_precision: 0.9550 - val_recall: 0.9545 - val_acc: 0.9547
Epoch 00020: early stopping
(tf2.3) quannm@quannm:/media/HDD/bkai$ 