(tf2.3) quannm@quannm:/media/HDD/bkai$ python main.py
2021-11-06 17:26:24.017098: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Amount of images: 1000
Training: 800 - Validation: 200
100 - 26
2021-11-06 17:26:25.182297: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2021-11-06 17:26:25.228095: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.228318: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 17:26:25.228356: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 17:26:25.230766: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 17:26:25.232553: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 17:26:25.232898: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 17:26:25.235077: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 17:26:25.236215: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 17:26:25.240084: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 17:26:25.240222: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.240430: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.240584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 17:26:25.240905: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-06 17:26:25.246887: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3199980000 Hz
2021-11-06 17:26:25.247264: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558b7ce7fe20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-11-06 17:26:25.247288: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-11-06 17:26:25.289686: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.289874: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558b7ed42270 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-11-06 17:26:25.289890: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce GTX 1070 Ti, Compute Capability 6.1
2021-11-06 17:26:25.290035: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.290138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 17:26:25.290157: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 17:26:25.290183: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 17:26:25.290195: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 17:26:25.290206: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 17:26:25.290216: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 17:26:25.290227: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 17:26:25.290239: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 17:26:25.290282: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.290398: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.290480: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 17:26:25.290502: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 17:26:25.565221: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-06 17:26:25.565252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2021-11-06 17:26:25.565259: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2021-11-06 17:26:25.565456: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.565700: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 17:26:25.565823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6264 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Using unet_att_eff0 model 
Model: "Attention-EfficientB0-Unet"
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
conv2d (Conv2D)                 (None, 32, 32, 256)  131328      conv2d_transpose[0][0]           
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 256)  553216      block4a_expand_activation[0][0]  
__________________________________________________________________________________________________
add (Add)                       (None, 32, 32, 256)  0           conv2d[0][0]                     
                                                                 conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation (Activation)         (None, 32, 32, 256)  0           add[0][0]                        
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 1)    257         activation[0][0]                 
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 1)    0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 32, 32, 1)    0           activation_1[0][0]               
__________________________________________________________________________________________________
lambda (Lambda)                 (None, 32, 32, 240)  0           up_sampling2d[0][0]              
__________________________________________________________________________________________________
multiply (Multiply)             (None, 32, 32, 240)  0           lambda[0][0]                     
                                                                 block4a_expand_activation[0][0]  
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 240)  57840       multiply[0][0]                   
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 32, 32, 240)  960         conv2d_3[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 32, 32, 752)  0           conv2d_transpose[0][0]           
                                                                 batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 512)  3465728     concatenate[0][0]                
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 512)  2048        conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 512)  0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 512)  2359808     activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 512)  2048        conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 512)  0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 256)  524544      activation_3[0][0]               
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 64, 64, 128)  32896       conv2d_transpose_1[0][0]         
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 64, 64, 128)  166016      block3a_expand_activation[0][0]  
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 64, 128)  0           conv2d_6[0][0]                   
                                                                 conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 64, 64, 128)  0           add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 64, 64, 1)    129         activation_4[0][0]               
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 64, 64, 1)    0           conv2d_8[0][0]                   
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 64, 64, 1)    0           activation_5[0][0]               
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 64, 64, 144)  0           up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 64, 64, 144)  0           lambda_1[0][0]                   
                                                                 block3a_expand_activation[0][0]  
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 64, 64, 144)  20880       multiply_1[0][0]                 
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 64, 64, 144)  576         conv2d_9[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 64, 64, 400)  0           conv2d_transpose_1[0][0]         
                                                                 batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 64, 64, 256)  921856      concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 64, 64, 256)  1024        conv2d_10[0][0]                  
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 64, 64, 256)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 64, 64, 256)  590080      activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 64, 64, 256)  1024        conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 64, 64, 256)  0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 128, 128, 128 131200      activation_7[0][0]               
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 128, 128, 64) 8256        conv2d_transpose_2[0][0]         
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 128, 128, 64) 55360       block2a_expand_activation[0][0]  
__________________________________________________________________________________________________
add_2 (Add)                     (None, 128, 128, 64) 0           conv2d_12[0][0]                  
                                                                 conv2d_13[0][0]                  
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 128, 128, 64) 0           add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 128, 128, 1)  65          activation_8[0][0]               
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 128, 128, 1)  0           conv2d_14[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 128, 128, 1)  0           activation_9[0][0]               
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 128, 128, 96) 0           up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
multiply_2 (Multiply)           (None, 128, 128, 96) 0           lambda_2[0][0]                   
                                                                 block2a_expand_activation[0][0]  
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 128, 128, 96) 9312        multiply_2[0][0]                 
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 128, 128, 96) 384         conv2d_15[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 128, 128, 224 0           conv2d_transpose_2[0][0]         
                                                                 batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 128, 128, 128 258176      concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 128, 128, 128 512         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 128, 128, 128 0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 128, 128, 128 147584      activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 128, 128, 128 512         conv2d_17[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 128, 128, 128 0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 256, 256, 64) 32832       activation_11[0][0]              
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 256, 256, 32) 2080        conv2d_transpose_3[0][0]         
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 256, 256, 32) 896         input[0][0]                      
__________________________________________________________________________________________________
add_3 (Add)                     (None, 256, 256, 32) 0           conv2d_18[0][0]                  
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 256, 256, 32) 0           add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 256, 256, 1)  33          activation_12[0][0]              
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 256, 256, 1)  0           conv2d_20[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 256, 256, 1)  0           activation_13[0][0]              
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, 256, 256, 3)  0           up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
multiply_3 (Multiply)           (None, 256, 256, 3)  0           lambda_3[0][0]                   
                                                                 input[0][0]                      
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 256, 256, 3)  12          multiply_3[0][0]                 
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 256, 256, 3)  12          conv2d_21[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256, 256, 67) 0           conv2d_transpose_3[0][0]         
                                                                 batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 256, 256, 64) 38656       concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 256, 256, 64) 256         conv2d_22[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 256, 256, 64) 0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 256, 256, 64) 36928       activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 256, 256, 64) 256         conv2d_23[0][0]                  
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 256, 256, 64) 0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 256, 256, 3)  195         activation_15[0][0]              
==================================================================================================
Total params: 11,878,470
Trainable params: 11,856,153
Non-trainable params: 22,317
__________________________________________________________________________________________________
Epoch 1/20
2021-11-06 17:26:32.534362: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 17:26:33.032361: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
100/100 [==============================] - ETA: 0s - loss: 0.4277 - precision: 0.9442 - recall: 0.8169 - acc: 0.8933 
Epoch 00001: saving model to unet_att_eff0.h5
100/100 [==============================] - 41s 412ms/step - loss: 0.4277 - precision: 0.9442 - recall: 0.8169 - acc: 0.8933 - val_loss: 1.1454 - val_precision: 0.3210 - val_recall: 0.1804 - val_acc: 0.4151
Epoch 2/20
100/100 [==============================] - ETA: 0s - loss: 0.1676 - precision: 0.9767 - recall: 0.9709 - acc: 0.9739 
Epoch 00002: saving model to unet_att_eff0.h5
100/100 [==============================] - 40s 401ms/step - loss: 0.1676 - precision: 0.9767 - recall: 0.9709 - acc: 0.9739 - val_loss: 0.3448 - val_precision: 0.9481 - val_recall: 0.9459 - val_acc: 0.9476
Epoch 3/20
100/100 [==============================] - ETA: 0s - loss: 0.1307 - precision: 0.9796 - recall: 0.9745 - acc: 0.9770 
Epoch 00003: saving model to unet_att_eff0.h5
100/100 [==============================] - 41s 415ms/step - loss: 0.1307 - precision: 0.9796 - recall: 0.9745 - acc: 0.9770 - val_loss: 0.5966 - val_precision: 0.7700 - val_recall: 0.7221 - val_acc: 0.7491
Epoch 4/20
100/100 [==============================] - ETA: 0s - loss: 0.1108 - precision: 0.9820 - recall: 0.9782 - acc: 0.9800 
Epoch 00004: saving model to unet_att_eff0.h5
100/100 [==============================] - 40s 404ms/step - loss: 0.1108 - precision: 0.9820 - recall: 0.9782 - acc: 0.9800 - val_loss: 0.2939 - val_precision: 0.9296 - val_recall: 0.9248 - val_acc: 0.9273
Epoch 5/20
100/100 [==============================] - ETA: 0s - loss: 0.0919 - precision: 0.9844 - recall: 0.9806 - acc: 0.9824 
Epoch 00005: saving model to unet_att_eff0.h5
100/100 [==============================] - 43s 427ms/step - loss: 0.0919 - precision: 0.9844 - recall: 0.9806 - acc: 0.9824 - val_loss: 0.2132 - val_precision: 0.9519 - val_recall: 0.9502 - val_acc: 0.9511
Epoch 6/20
100/100 [==============================] - ETA: 0s - loss: 0.0790 - precision: 0.9865 - recall: 0.9826 - acc: 0.9844 
Epoch 00006: saving model to unet_att_eff0.h5
100/100 [==============================] - 43s 428ms/step - loss: 0.0790 - precision: 0.9865 - recall: 0.9826 - acc: 0.9844 - val_loss: 0.1572 - val_precision: 0.9633 - val_recall: 0.9599 - val_acc: 0.9617
Epoch 7/20
100/100 [==============================] - ETA: 0s - loss: 0.0736 - precision: 0.9866 - recall: 0.9819 - acc: 0.9840 
Epoch 00007: saving model to unet_att_eff0.h5
100/100 [==============================] - 43s 433ms/step - loss: 0.0736 - precision: 0.9866 - recall: 0.9819 - acc: 0.9840 - val_loss: 0.1463 - val_precision: 0.9662 - val_recall: 0.9621 - val_acc: 0.9642
Epoch 8/20
100/100 [==============================] - ETA: 0s - loss: 0.0621 - precision: 0.9891 - recall: 0.9841 - acc: 0.9862 
Epoch 00008: saving model to unet_att_eff0.h5
100/100 [==============================] - 44s 435ms/step - loss: 0.0621 - precision: 0.9891 - recall: 0.9841 - acc: 0.9862 - val_loss: 0.1955 - val_precision: 0.9565 - val_recall: 0.9476 - val_acc: 0.9524
Epoch 9/20
100/100 [==============================] - ETA: 0s - loss: 0.0566 - precision: 0.9900 - recall: 0.9846 - acc: 0.9872 
Epoch 00009: saving model to unet_att_eff0.h5
100/100 [==============================] - 42s 417ms/step - loss: 0.0566 - precision: 0.9900 - recall: 0.9846 - acc: 0.9872 - val_loss: 0.1095 - val_precision: 0.9772 - val_recall: 0.9680 - val_acc: 0.9725
Epoch 10/20
100/100 [==============================] - ETA: 0s - loss: 0.0515 - precision: 0.9903 - recall: 0.9864 - acc: 0.9883 
Epoch 00010: saving model to unet_att_eff0.h5
100/100 [==============================] - 42s 415ms/step - loss: 0.0515 - precision: 0.9903 - recall: 0.9864 - acc: 0.9883 - val_loss: 0.1398 - val_precision: 0.9695 - val_recall: 0.9576 - val_acc: 0.9622
Epoch 11/20
100/100 [==============================] - ETA: 0s - loss: 0.0461 - precision: 0.9908 - recall: 0.9881 - acc: 0.9894 
Epoch 00011: saving model to unet_att_eff0.h5
100/100 [==============================] - 42s 423ms/step - loss: 0.0461 - precision: 0.9908 - recall: 0.9881 - acc: 0.9894 - val_loss: 0.1221 - val_precision: 0.9698 - val_recall: 0.9640 - val_acc: 0.9662
Epoch 12/20
100/100 [==============================] - ETA: 0s - loss: 0.0439 - precision: 0.9904 - recall: 0.9880 - acc: 0.9891 
Epoch 00012: saving model to unet_att_eff0.h5
100/100 [==============================] - 41s 411ms/step - loss: 0.0439 - precision: 0.9904 - recall: 0.9880 - acc: 0.9891 - val_loss: 0.2855 - val_precision: 0.9287 - val_recall: 0.9228 - val_acc: 0.9253
Epoch 13/20
100/100 [==============================] - ETA: 0s - loss: 0.0399 - precision: 0.9914 - recall: 0.9894 - acc: 0.9903 
Epoch 00013: saving model to unet_att_eff0.h5
100/100 [==============================] - 42s 417ms/step - loss: 0.0399 - precision: 0.9914 - recall: 0.9894 - acc: 0.9903 - val_loss: 0.2672 - val_precision: 0.9207 - val_recall: 0.9027 - val_acc: 0.9113
Epoch 14/20
100/100 [==============================] - ETA: 0s - loss: 0.0351 - precision: 0.9922 - recall: 0.9910 - acc: 0.9915 
Epoch 00014: saving model to unet_att_eff0.h5

Epoch 00014: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-06.
100/100 [==============================] - 41s 409ms/step - loss: 0.0351 - precision: 0.9922 - recall: 0.9910 - acc: 0.9915 - val_loss: 0.1723 - val_precision: 0.9530 - val_recall: 0.9469 - val_acc: 0.9499
Epoch 00014: early stopping