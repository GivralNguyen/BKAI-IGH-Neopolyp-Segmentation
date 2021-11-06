(tf2.3) quannm@quannm:/media/HDD/bkai$ python main.py
2021-11-06 15:24:16.462820: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Amount of images: 1000
Training: 800 - Validation: 200
100 - 26
2021-11-06 15:24:17.663268: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2021-11-06 15:24:17.691880: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:17.692158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 15:24:17.692180: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:24:17.693527: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 15:24:17.694800: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 15:24:17.695052: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 15:24:17.696599: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 15:24:17.697374: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 15:24:17.700221: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:24:17.700347: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:17.700525: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:17.700646: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 15:24:17.700967: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-06 15:24:17.705725: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3199980000 Hz
2021-11-06 15:24:17.706041: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564be79fd970 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-11-06 15:24:17.706060: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-11-06 15:24:17.759975: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:17.760167: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564be98bf200 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-11-06 15:24:17.760182: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce GTX 1070 Ti, Compute Capability 6.1
2021-11-06 15:24:17.760346: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:17.760450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 15:24:17.760472: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:24:17.760499: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 15:24:17.760514: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 15:24:17.760525: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 15:24:17.760536: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 15:24:17.760547: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 15:24:17.760558: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:24:17.760603: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:17.760723: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:17.760826: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 15:24:17.760855: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:24:18.041667: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-06 15:24:18.041695: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2021-11-06 15:24:18.041701: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2021-11-06 15:24:18.041852: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:18.042017: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:24:18.042127: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6021 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Using att_unet_mb2 model 
WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
Model: "MobilenetV2_Unet"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input (InputLayer)              [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
Conv1_pad (ZeroPadding2D)       (None, 257, 257, 3)  0           input[0][0]                      
__________________________________________________________________________________________________
Conv1 (Conv2D)                  (None, 128, 128, 32) 864         Conv1_pad[0][0]                  
__________________________________________________________________________________________________
bn_Conv1 (BatchNormalization)   (None, 128, 128, 32) 128         Conv1[0][0]                      
__________________________________________________________________________________________________
Conv1_relu (ReLU)               (None, 128, 128, 32) 0           bn_Conv1[0][0]                   
__________________________________________________________________________________________________
expanded_conv_depthwise (Depthw (None, 128, 128, 32) 288         Conv1_relu[0][0]                 
__________________________________________________________________________________________________
expanded_conv_depthwise_BN (Bat (None, 128, 128, 32) 128         expanded_conv_depthwise[0][0]    
__________________________________________________________________________________________________
expanded_conv_depthwise_relu (R (None, 128, 128, 32) 0           expanded_conv_depthwise_BN[0][0] 
__________________________________________________________________________________________________
expanded_conv_project (Conv2D)  (None, 128, 128, 16) 512         expanded_conv_depthwise_relu[0][0
__________________________________________________________________________________________________
expanded_conv_project_BN (Batch (None, 128, 128, 16) 64          expanded_conv_project[0][0]      
__________________________________________________________________________________________________
block_1_expand (Conv2D)         (None, 128, 128, 96) 1536        expanded_conv_project_BN[0][0]   
__________________________________________________________________________________________________
block_1_expand_BN (BatchNormali (None, 128, 128, 96) 384         block_1_expand[0][0]             
__________________________________________________________________________________________________
block_1_expand_relu (ReLU)      (None, 128, 128, 96) 0           block_1_expand_BN[0][0]          
__________________________________________________________________________________________________
block_1_pad (ZeroPadding2D)     (None, 129, 129, 96) 0           block_1_expand_relu[0][0]        
__________________________________________________________________________________________________
block_1_depthwise (DepthwiseCon (None, 64, 64, 96)   864         block_1_pad[0][0]                
__________________________________________________________________________________________________
block_1_depthwise_BN (BatchNorm (None, 64, 64, 96)   384         block_1_depthwise[0][0]          
__________________________________________________________________________________________________
block_1_depthwise_relu (ReLU)   (None, 64, 64, 96)   0           block_1_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_1_project (Conv2D)        (None, 64, 64, 24)   2304        block_1_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_1_project_BN (BatchNormal (None, 64, 64, 24)   96          block_1_project[0][0]            
__________________________________________________________________________________________________
block_2_expand (Conv2D)         (None, 64, 64, 144)  3456        block_1_project_BN[0][0]         
__________________________________________________________________________________________________
block_2_expand_BN (BatchNormali (None, 64, 64, 144)  576         block_2_expand[0][0]             
__________________________________________________________________________________________________
block_2_expand_relu (ReLU)      (None, 64, 64, 144)  0           block_2_expand_BN[0][0]          
__________________________________________________________________________________________________
block_2_depthwise (DepthwiseCon (None, 64, 64, 144)  1296        block_2_expand_relu[0][0]        
__________________________________________________________________________________________________
block_2_depthwise_BN (BatchNorm (None, 64, 64, 144)  576         block_2_depthwise[0][0]          
__________________________________________________________________________________________________
block_2_depthwise_relu (ReLU)   (None, 64, 64, 144)  0           block_2_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_2_project (Conv2D)        (None, 64, 64, 24)   3456        block_2_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_2_project_BN (BatchNormal (None, 64, 64, 24)   96          block_2_project[0][0]            
__________________________________________________________________________________________________
block_2_add (Add)               (None, 64, 64, 24)   0           block_1_project_BN[0][0]         
                                                                 block_2_project_BN[0][0]         
__________________________________________________________________________________________________
block_3_expand (Conv2D)         (None, 64, 64, 144)  3456        block_2_add[0][0]                
__________________________________________________________________________________________________
block_3_expand_BN (BatchNormali (None, 64, 64, 144)  576         block_3_expand[0][0]             
__________________________________________________________________________________________________
block_3_expand_relu (ReLU)      (None, 64, 64, 144)  0           block_3_expand_BN[0][0]          
__________________________________________________________________________________________________
block_3_pad (ZeroPadding2D)     (None, 65, 65, 144)  0           block_3_expand_relu[0][0]        
__________________________________________________________________________________________________
block_3_depthwise (DepthwiseCon (None, 32, 32, 144)  1296        block_3_pad[0][0]                
__________________________________________________________________________________________________
block_3_depthwise_BN (BatchNorm (None, 32, 32, 144)  576         block_3_depthwise[0][0]          
__________________________________________________________________________________________________
block_3_depthwise_relu (ReLU)   (None, 32, 32, 144)  0           block_3_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_3_project (Conv2D)        (None, 32, 32, 32)   4608        block_3_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_3_project_BN (BatchNormal (None, 32, 32, 32)   128         block_3_project[0][0]            
__________________________________________________________________________________________________
block_4_expand (Conv2D)         (None, 32, 32, 192)  6144        block_3_project_BN[0][0]         
__________________________________________________________________________________________________
block_4_expand_BN (BatchNormali (None, 32, 32, 192)  768         block_4_expand[0][0]             
__________________________________________________________________________________________________
block_4_expand_relu (ReLU)      (None, 32, 32, 192)  0           block_4_expand_BN[0][0]          
__________________________________________________________________________________________________
block_4_depthwise (DepthwiseCon (None, 32, 32, 192)  1728        block_4_expand_relu[0][0]        
__________________________________________________________________________________________________
block_4_depthwise_BN (BatchNorm (None, 32, 32, 192)  768         block_4_depthwise[0][0]          
__________________________________________________________________________________________________
block_4_depthwise_relu (ReLU)   (None, 32, 32, 192)  0           block_4_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_4_project (Conv2D)        (None, 32, 32, 32)   6144        block_4_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_4_project_BN (BatchNormal (None, 32, 32, 32)   128         block_4_project[0][0]            
__________________________________________________________________________________________________
block_4_add (Add)               (None, 32, 32, 32)   0           block_3_project_BN[0][0]         
                                                                 block_4_project_BN[0][0]         
__________________________________________________________________________________________________
block_5_expand (Conv2D)         (None, 32, 32, 192)  6144        block_4_add[0][0]                
__________________________________________________________________________________________________
block_5_expand_BN (BatchNormali (None, 32, 32, 192)  768         block_5_expand[0][0]             
__________________________________________________________________________________________________
block_5_expand_relu (ReLU)      (None, 32, 32, 192)  0           block_5_expand_BN[0][0]          
__________________________________________________________________________________________________
block_5_depthwise (DepthwiseCon (None, 32, 32, 192)  1728        block_5_expand_relu[0][0]        
__________________________________________________________________________________________________
block_5_depthwise_BN (BatchNorm (None, 32, 32, 192)  768         block_5_depthwise[0][0]          
__________________________________________________________________________________________________
block_5_depthwise_relu (ReLU)   (None, 32, 32, 192)  0           block_5_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_5_project (Conv2D)        (None, 32, 32, 32)   6144        block_5_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_5_project_BN (BatchNormal (None, 32, 32, 32)   128         block_5_project[0][0]            
__________________________________________________________________________________________________
block_5_add (Add)               (None, 32, 32, 32)   0           block_4_add[0][0]                
                                                                 block_5_project_BN[0][0]         
__________________________________________________________________________________________________
block_6_expand (Conv2D)         (None, 32, 32, 192)  6144        block_5_add[0][0]                
__________________________________________________________________________________________________
block_6_expand_BN (BatchNormali (None, 32, 32, 192)  768         block_6_expand[0][0]             
__________________________________________________________________________________________________
block_6_expand_relu (ReLU)      (None, 32, 32, 192)  0           block_6_expand_BN[0][0]          
__________________________________________________________________________________________________
block_6_pad (ZeroPadding2D)     (None, 33, 33, 192)  0           block_6_expand_relu[0][0]        
__________________________________________________________________________________________________
block_6_depthwise (DepthwiseCon (None, 16, 16, 192)  1728        block_6_pad[0][0]                
__________________________________________________________________________________________________
block_6_depthwise_BN (BatchNorm (None, 16, 16, 192)  768         block_6_depthwise[0][0]          
__________________________________________________________________________________________________
block_6_depthwise_relu (ReLU)   (None, 16, 16, 192)  0           block_6_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_6_project (Conv2D)        (None, 16, 16, 64)   12288       block_6_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_6_project_BN (BatchNormal (None, 16, 16, 64)   256         block_6_project[0][0]            
__________________________________________________________________________________________________
block_7_expand (Conv2D)         (None, 16, 16, 384)  24576       block_6_project_BN[0][0]         
__________________________________________________________________________________________________
block_7_expand_BN (BatchNormali (None, 16, 16, 384)  1536        block_7_expand[0][0]             
__________________________________________________________________________________________________
block_7_expand_relu (ReLU)      (None, 16, 16, 384)  0           block_7_expand_BN[0][0]          
__________________________________________________________________________________________________
block_7_depthwise (DepthwiseCon (None, 16, 16, 384)  3456        block_7_expand_relu[0][0]        
__________________________________________________________________________________________________
block_7_depthwise_BN (BatchNorm (None, 16, 16, 384)  1536        block_7_depthwise[0][0]          
__________________________________________________________________________________________________
block_7_depthwise_relu (ReLU)   (None, 16, 16, 384)  0           block_7_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_7_project (Conv2D)        (None, 16, 16, 64)   24576       block_7_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_7_project_BN (BatchNormal (None, 16, 16, 64)   256         block_7_project[0][0]            
__________________________________________________________________________________________________
block_7_add (Add)               (None, 16, 16, 64)   0           block_6_project_BN[0][0]         
                                                                 block_7_project_BN[0][0]         
__________________________________________________________________________________________________
block_8_expand (Conv2D)         (None, 16, 16, 384)  24576       block_7_add[0][0]                
__________________________________________________________________________________________________
block_8_expand_BN (BatchNormali (None, 16, 16, 384)  1536        block_8_expand[0][0]             
__________________________________________________________________________________________________
block_8_expand_relu (ReLU)      (None, 16, 16, 384)  0           block_8_expand_BN[0][0]          
__________________________________________________________________________________________________
block_8_depthwise (DepthwiseCon (None, 16, 16, 384)  3456        block_8_expand_relu[0][0]        
__________________________________________________________________________________________________
block_8_depthwise_BN (BatchNorm (None, 16, 16, 384)  1536        block_8_depthwise[0][0]          
__________________________________________________________________________________________________
block_8_depthwise_relu (ReLU)   (None, 16, 16, 384)  0           block_8_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_8_project (Conv2D)        (None, 16, 16, 64)   24576       block_8_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_8_project_BN (BatchNormal (None, 16, 16, 64)   256         block_8_project[0][0]            
__________________________________________________________________________________________________
block_8_add (Add)               (None, 16, 16, 64)   0           block_7_add[0][0]                
                                                                 block_8_project_BN[0][0]         
__________________________________________________________________________________________________
block_9_expand (Conv2D)         (None, 16, 16, 384)  24576       block_8_add[0][0]                
__________________________________________________________________________________________________
block_9_expand_BN (BatchNormali (None, 16, 16, 384)  1536        block_9_expand[0][0]             
__________________________________________________________________________________________________
block_9_expand_relu (ReLU)      (None, 16, 16, 384)  0           block_9_expand_BN[0][0]          
__________________________________________________________________________________________________
block_9_depthwise (DepthwiseCon (None, 16, 16, 384)  3456        block_9_expand_relu[0][0]        
__________________________________________________________________________________________________
block_9_depthwise_BN (BatchNorm (None, 16, 16, 384)  1536        block_9_depthwise[0][0]          
__________________________________________________________________________________________________
block_9_depthwise_relu (ReLU)   (None, 16, 16, 384)  0           block_9_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_9_project (Conv2D)        (None, 16, 16, 64)   24576       block_9_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_9_project_BN (BatchNormal (None, 16, 16, 64)   256         block_9_project[0][0]            
__________________________________________________________________________________________________
block_9_add (Add)               (None, 16, 16, 64)   0           block_8_add[0][0]                
                                                                 block_9_project_BN[0][0]         
__________________________________________________________________________________________________
block_10_expand (Conv2D)        (None, 16, 16, 384)  24576       block_9_add[0][0]                
__________________________________________________________________________________________________
block_10_expand_BN (BatchNormal (None, 16, 16, 384)  1536        block_10_expand[0][0]            
__________________________________________________________________________________________________
block_10_expand_relu (ReLU)     (None, 16, 16, 384)  0           block_10_expand_BN[0][0]         
__________________________________________________________________________________________________
block_10_depthwise (DepthwiseCo (None, 16, 16, 384)  3456        block_10_expand_relu[0][0]       
__________________________________________________________________________________________________
block_10_depthwise_BN (BatchNor (None, 16, 16, 384)  1536        block_10_depthwise[0][0]         
__________________________________________________________________________________________________
block_10_depthwise_relu (ReLU)  (None, 16, 16, 384)  0           block_10_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_10_project (Conv2D)       (None, 16, 16, 96)   36864       block_10_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_10_project_BN (BatchNorma (None, 16, 16, 96)   384         block_10_project[0][0]           
__________________________________________________________________________________________________
block_11_expand (Conv2D)        (None, 16, 16, 576)  55296       block_10_project_BN[0][0]        
__________________________________________________________________________________________________
block_11_expand_BN (BatchNormal (None, 16, 16, 576)  2304        block_11_expand[0][0]            
__________________________________________________________________________________________________
block_11_expand_relu (ReLU)     (None, 16, 16, 576)  0           block_11_expand_BN[0][0]         
__________________________________________________________________________________________________
block_11_depthwise (DepthwiseCo (None, 16, 16, 576)  5184        block_11_expand_relu[0][0]       
__________________________________________________________________________________________________
block_11_depthwise_BN (BatchNor (None, 16, 16, 576)  2304        block_11_depthwise[0][0]         
__________________________________________________________________________________________________
block_11_depthwise_relu (ReLU)  (None, 16, 16, 576)  0           block_11_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_11_project (Conv2D)       (None, 16, 16, 96)   55296       block_11_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_11_project_BN (BatchNorma (None, 16, 16, 96)   384         block_11_project[0][0]           
__________________________________________________________________________________________________
block_11_add (Add)              (None, 16, 16, 96)   0           block_10_project_BN[0][0]        
                                                                 block_11_project_BN[0][0]        
__________________________________________________________________________________________________
block_12_expand (Conv2D)        (None, 16, 16, 576)  55296       block_11_add[0][0]               
__________________________________________________________________________________________________
block_12_expand_BN (BatchNormal (None, 16, 16, 576)  2304        block_12_expand[0][0]            
__________________________________________________________________________________________________
block_12_expand_relu (ReLU)     (None, 16, 16, 576)  0           block_12_expand_BN[0][0]         
__________________________________________________________________________________________________
block_12_depthwise (DepthwiseCo (None, 16, 16, 576)  5184        block_12_expand_relu[0][0]       
__________________________________________________________________________________________________
block_12_depthwise_BN (BatchNor (None, 16, 16, 576)  2304        block_12_depthwise[0][0]         
__________________________________________________________________________________________________
block_12_depthwise_relu (ReLU)  (None, 16, 16, 576)  0           block_12_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_12_project (Conv2D)       (None, 16, 16, 96)   55296       block_12_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_12_project_BN (BatchNorma (None, 16, 16, 96)   384         block_12_project[0][0]           
__________________________________________________________________________________________________
block_12_add (Add)              (None, 16, 16, 96)   0           block_11_add[0][0]               
                                                                 block_12_project_BN[0][0]        
__________________________________________________________________________________________________
block_13_expand (Conv2D)        (None, 16, 16, 576)  55296       block_12_add[0][0]               
__________________________________________________________________________________________________
block_13_expand_BN (BatchNormal (None, 16, 16, 576)  2304        block_13_expand[0][0]            
__________________________________________________________________________________________________
block_13_expand_relu (ReLU)     (None, 16, 16, 576)  0           block_13_expand_BN[0][0]         
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 32, 32, 512)  1180160     block_13_expand_relu[0][0]       
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 32, 32, 256)  131328      conv2d_transpose[0][0]           
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 256)  442624      block_6_expand_relu[0][0]        
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
lambda (Lambda)                 (None, 32, 32, 192)  0           up_sampling2d[0][0]              
__________________________________________________________________________________________________
multiply (Multiply)             (None, 32, 32, 192)  0           lambda[0][0]                     
                                                                 block_6_expand_relu[0][0]        
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 192)  37056       multiply[0][0]                   
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 32, 32, 192)  768         conv2d_3[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 32, 32, 704)  0           conv2d_transpose[0][0]           
                                                                 batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 512)  3244544     concatenate[0][0]                
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
conv2d_7 (Conv2D)               (None, 64, 64, 128)  166016      block_3_expand_relu[0][0]        
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
                                                                 block_3_expand_relu[0][0]        
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
conv2d_13 (Conv2D)              (None, 128, 128, 64) 55360       block_1_expand_relu[0][0]        
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
                                                                 block_1_expand_relu[0][0]        
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
Total params: 10,999,439
Trainable params: 10,977,449
Non-trainable params: 21,990
__________________________________________________________________________________________________
Epoch 1/40
2021-11-06 15:24:23.346309: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:24:23.854599: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
100/100 [==============================] - ETA: 0s - loss: 0.6633 - precision: 0.9004 - recall: 0.6982 - acc: 0.8021 
Epoch 00001: saving model to att_unet_mb2.h5
100/100 [==============================] - 38s 378ms/step - loss: 0.6633 - precision: 0.9004 - recall: 0.6982 - acc: 0.8021 - val_loss: 0.8125 - val_precision: 0.9551 - val_recall: 0.0121 - val_acc: 0.9693
Epoch 2/40
100/100 [==============================] - ETA: 0s - loss: 0.2684 - precision: 0.9777 - recall: 0.9607 - acc: 0.9706 
Epoch 00002: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 369ms/step - loss: 0.2684 - precision: 0.9777 - recall: 0.9607 - acc: 0.9706 - val_loss: 0.4475 - val_precision: 0.9812 - val_recall: 0.9603 - val_acc: 0.9729
Epoch 3/40
100/100 [==============================] - ETA: 0s - loss: 0.2052 - precision: 0.9810 - recall: 0.9698 - acc: 0.9759 
Epoch 00003: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 372ms/step - loss: 0.2052 - precision: 0.9810 - recall: 0.9698 - acc: 0.9759 - val_loss: 0.2698 - val_precision: 0.9829 - val_recall: 0.9706 - val_acc: 0.9775
Epoch 4/40
100/100 [==============================] - ETA: 0s - loss: 0.1647 - precision: 0.9829 - recall: 0.9751 - acc: 0.9792 
Epoch 00004: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 366ms/step - loss: 0.1647 - precision: 0.9829 - recall: 0.9751 - acc: 0.9792 - val_loss: 0.2049 - val_precision: 0.9800 - val_recall: 0.9658 - val_acc: 0.9732
Epoch 5/40
100/100 [==============================] - ETA: 0s - loss: 0.1348 - precision: 0.9847 - recall: 0.9786 - acc: 0.9817 
Epoch 00005: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 367ms/step - loss: 0.1348 - precision: 0.9847 - recall: 0.9786 - acc: 0.9817 - val_loss: 0.1597 - val_precision: 0.9794 - val_recall: 0.9678 - val_acc: 0.9739
Epoch 6/40
100/100 [==============================] - ETA: 0s - loss: 0.1109 - precision: 0.9868 - recall: 0.9819 - acc: 0.9842 
Epoch 00006: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 367ms/step - loss: 0.1109 - precision: 0.9868 - recall: 0.9819 - acc: 0.9842 - val_loss: 0.1246 - val_precision: 0.9805 - val_recall: 0.9743 - val_acc: 0.9774
Epoch 7/40
100/100 [==============================] - ETA: 0s - loss: 0.0904 - precision: 0.9880 - recall: 0.9839 - acc: 0.9858 
Epoch 00007: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 370ms/step - loss: 0.0904 - precision: 0.9880 - recall: 0.9839 - acc: 0.9858 - val_loss: 0.1125 - val_precision: 0.9820 - val_recall: 0.9739 - val_acc: 0.9782
Epoch 8/40
100/100 [==============================] - ETA: 0s - loss: 0.0731 - precision: 0.9895 - recall: 0.9858 - acc: 0.9875 
Epoch 00008: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 370ms/step - loss: 0.0731 - precision: 0.9895 - recall: 0.9858 - acc: 0.9875 - val_loss: 0.1037 - val_precision: 0.9832 - val_recall: 0.9773 - val_acc: 0.9803
Epoch 9/40
100/100 [==============================] - ETA: 0s - loss: 0.0621 - precision: 0.9906 - recall: 0.9875 - acc: 0.9889 
Epoch 00009: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 374ms/step - loss: 0.0621 - precision: 0.9906 - recall: 0.9875 - acc: 0.9889 - val_loss: 0.1117 - val_precision: 0.9815 - val_recall: 0.9735 - val_acc: 0.9774
Epoch 10/40
100/100 [==============================] - ETA: 0s - loss: 0.0538 - precision: 0.9917 - recall: 0.9895 - acc: 0.9905 
Epoch 00010: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 371ms/step - loss: 0.0538 - precision: 0.9917 - recall: 0.9895 - acc: 0.9905 - val_loss: 0.1233 - val_precision: 0.9776 - val_recall: 0.9693 - val_acc: 0.9731
Epoch 11/40
100/100 [==============================] - ETA: 0s - loss: 0.0491 - precision: 0.9918 - recall: 0.9902 - acc: 0.9910 
Epoch 00011: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 373ms/step - loss: 0.0491 - precision: 0.9918 - recall: 0.9902 - acc: 0.9910 - val_loss: 0.0894 - val_precision: 0.9828 - val_recall: 0.9766 - val_acc: 0.9794
Epoch 12/40
100/100 [==============================] - ETA: 0s - loss: 0.0433 - precision: 0.9926 - recall: 0.9915 - acc: 0.9921 
Epoch 00012: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 370ms/step - loss: 0.0433 - precision: 0.9926 - recall: 0.9915 - acc: 0.9921 - val_loss: 0.0703 - val_precision: 0.9848 - val_recall: 0.9831 - val_acc: 0.9839
Epoch 13/40
100/100 [==============================] - ETA: 0s - loss: 0.0379 - precision: 0.9937 - recall: 0.9930 - acc: 0.9933 
Epoch 00013: saving model to att_unet_mb2.h5
100/100 [==============================] - 38s 382ms/step - loss: 0.0379 - precision: 0.9937 - recall: 0.9930 - acc: 0.9933 - val_loss: 0.1072 - val_precision: 0.9768 - val_recall: 0.9730 - val_acc: 0.9747
Epoch 14/40
100/100 [==============================] - ETA: 0s - loss: 0.0312 - precision: 0.9950 - recall: 0.9945 - acc: 0.9948 
Epoch 00014: saving model to att_unet_mb2.h5
100/100 [==============================] - 38s 378ms/step - loss: 0.0312 - precision: 0.9950 - recall: 0.9945 - acc: 0.9948 - val_loss: 0.0843 - val_precision: 0.9791 - val_recall: 0.9771 - val_acc: 0.9780
Epoch 15/40
100/100 [==============================] - ETA: 0s - loss: 0.0293 - precision: 0.9950 - recall: 0.9945 - acc: 0.9947 
Epoch 00015: saving model to att_unet_mb2.h5
100/100 [==============================] - 38s 380ms/step - loss: 0.0293 - precision: 0.9950 - recall: 0.9945 - acc: 0.9947 - val_loss: 0.0649 - val_precision: 0.9853 - val_recall: 0.9842 - val_acc: 0.9847
Epoch 16/40
100/100 [==============================] - ETA: 0s - loss: 0.0252 - precision: 0.9958 - recall: 0.9955 - acc: 0.9956 
Epoch 00016: saving model to att_unet_mb2.h5
100/100 [==============================] - 38s 377ms/step - loss: 0.0252 - precision: 0.9958 - recall: 0.9955 - acc: 0.9956 - val_loss: 0.0809 - val_precision: 0.9796 - val_recall: 0.9782 - val_acc: 0.9788
Epoch 17/40
100/100 [==============================] - ETA: 0s - loss: 0.0226 - precision: 0.9962 - recall: 0.9959 - acc: 0.9961 
Epoch 00017: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 373ms/step - loss: 0.0226 - precision: 0.9962 - recall: 0.9959 - acc: 0.9961 - val_loss: 0.0801 - val_precision: 0.9817 - val_recall: 0.9807 - val_acc: 0.9811
Epoch 18/40
100/100 [==============================] - ETA: 0s - loss: 0.0210 - precision: 0.9963 - recall: 0.9961 - acc: 0.9962 
Epoch 00018: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 371ms/step - loss: 0.0210 - precision: 0.9963 - recall: 0.9961 - acc: 0.9962 - val_loss: 0.0678 - val_precision: 0.9854 - val_recall: 0.9847 - val_acc: 0.9850
Epoch 19/40
100/100 [==============================] - ETA: 0s - loss: 0.0196 - precision: 0.9965 - recall: 0.9962 - acc: 0.9963 
Epoch 00019: saving model to att_unet_mb2.h5
100/100 [==============================] - 37s 371ms/step - loss: 0.0196 - precision: 0.9965 - recall: 0.9962 - acc: 0.9963 - val_loss: 0.0712 - val_precision: 0.9835 - val_recall: 0.9824 - val_acc: 0.9829
Epoch 20/40
100/100 [==============================] - ETA: 0s - loss: 0.0185 - precision: 0.9965 - recall: 0.9963 - acc: 0.9964 
Epoch 00020: saving model to att_unet_mb2.h5

Epoch 00020: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-06.
100/100 [==============================] - 37s 373ms/step - loss: 0.0185 - precision: 0.9965 - recall: 0.9963 - acc: 0.9964 - val_loss: 0.1580 - val_precision: 0.9714 - val_recall: 0.9697 - val_acc: 0.9704
Epoch 00020: early stopping