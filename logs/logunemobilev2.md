(tf2.3) quannm@quannm:/media/HDD/bkai$ python main.py
2021-11-03 15:57:13.678767: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Amount of images: 1000
Training: 800 - Validation: 200
100 - 26
2021-11-03 15:57:14.876141: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2021-11-03 15:57:14.903299: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:14.903432: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-03 15:57:14.903450: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-03 15:57:14.904913: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-03 15:57:14.906082: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-03 15:57:14.906294: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-03 15:57:14.907467: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-03 15:57:14.908156: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-03 15:57:14.911272: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-03 15:57:14.911529: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:14.911715: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:14.911817: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-03 15:57:14.912063: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-03 15:57:14.916565: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3199980000 Hz
2021-11-03 15:57:14.916871: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56233f10d060 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-11-03 15:57:14.916884: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-11-03 15:57:14.968238: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:14.968438: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562341070250 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-11-03 15:57:14.968455: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce GTX 1070 Ti, Compute Capability 6.1
2021-11-03 15:57:14.968621: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:14.968759: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-03 15:57:14.968791: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-03 15:57:14.968834: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-03 15:57:14.968853: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-03 15:57:14.968866: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-03 15:57:14.968877: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-03 15:57:14.968888: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-03 15:57:14.968900: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-03 15:57:14.968954: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:14.969075: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:14.969163: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-03 15:57:14.969182: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-03 15:57:15.240457: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-03 15:57:15.240485: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2021-11-03 15:57:15.240491: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2021-11-03 15:57:15.240654: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:15.240844: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-03 15:57:15.240956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6653 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
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
concatenate (Concatenate)       (None, 32, 32, 704)  0           conv2d_transpose[0][0]           
                                                                 block_6_expand_relu[0][0]        
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 32, 32, 512)  3244544     concatenate[0][0]                
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
                                                                 block_3_expand_relu[0][0]        
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
                                                                 block_1_expand_relu[0][0]        
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
Total params: 10,090,499
Trainable params: 10,069,379
Non-trainable params: 21,120
__________________________________________________________________________________________________
Epoch 1/20
2021-11-03 15:57:19.557438: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-03 15:57:20.031951: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
100/100 [==============================] - ETA: 0s - loss: 0.4694 - precision: 0.9054 - recall: 0.7861 - acc: 0.8589 
Epoch 00001: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 322ms/step - loss: 0.4694 - precision: 0.9054 - recall: 0.7861 - acc: 0.8589 - val_loss: 2.0780 - val_precision: 0.4978 - val_recall: 0.2967 - val_acc: 0.4934
Epoch 2/20
100/100 [==============================] - ETA: 0s - loss: 0.1701 - precision: 0.9797 - recall: 0.9734 - acc: 0.9767 
Epoch 00002: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 314ms/step - loss: 0.1701 - precision: 0.9797 - recall: 0.9734 - acc: 0.9767 - val_loss: 0.2934 - val_precision: 0.9831 - val_recall: 0.9674 - val_acc: 0.9762
Epoch 3/20
100/100 [==============================] - ETA: 0s - loss: 0.1327 - precision: 0.9839 - recall: 0.9800 - acc: 0.9819 
Epoch 00003: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 315ms/step - loss: 0.1327 - precision: 0.9839 - recall: 0.9800 - acc: 0.9819 - val_loss: 0.1576 - val_precision: 0.9820 - val_recall: 0.9745 - val_acc: 0.9784
Epoch 4/20
100/100 [==============================] - ETA: 0s - loss: 0.1088 - precision: 0.9866 - recall: 0.9832 - acc: 0.9849 
Epoch 00004: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 311ms/step - loss: 0.1088 - precision: 0.9866 - recall: 0.9832 - acc: 0.9849 - val_loss: 0.1266 - val_precision: 0.9830 - val_recall: 0.9748 - val_acc: 0.9791
Epoch 5/20
100/100 [==============================] - ETA: 0s - loss: 0.0910 - precision: 0.9884 - recall: 0.9854 - acc: 0.9867 
Epoch 00005: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 311ms/step - loss: 0.0910 - precision: 0.9884 - recall: 0.9854 - acc: 0.9867 - val_loss: 0.1088 - val_precision: 0.9820 - val_recall: 0.9758 - val_acc: 0.9789
Epoch 6/20
100/100 [==============================] - ETA: 0s - loss: 0.0793 - precision: 0.9890 - recall: 0.9856 - acc: 0.9871 
Epoch 00006: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 314ms/step - loss: 0.0793 - precision: 0.9890 - recall: 0.9856 - acc: 0.9871 - val_loss: 0.1263 - val_precision: 0.9801 - val_recall: 0.9629 - val_acc: 0.9711
Epoch 7/20
100/100 [==============================] - ETA: 0s - loss: 0.0670 - precision: 0.9910 - recall: 0.9878 - acc: 0.9892 
Epoch 00007: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 319ms/step - loss: 0.0670 - precision: 0.9910 - recall: 0.9878 - acc: 0.9892 - val_loss: 0.0902 - val_precision: 0.9830 - val_recall: 0.9736 - val_acc: 0.9781
Epoch 8/20
100/100 [==============================] - ETA: 0s - loss: 0.0573 - precision: 0.9924 - recall: 0.9901 - acc: 0.9912 
Epoch 00008: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 318ms/step - loss: 0.0573 - precision: 0.9924 - recall: 0.9901 - acc: 0.9912 - val_loss: 0.1288 - val_precision: 0.9760 - val_recall: 0.9620 - val_acc: 0.9682
Epoch 9/20
100/100 [==============================] - ETA: 0s - loss: 0.0510 - precision: 0.9927 - recall: 0.9908 - acc: 0.9918 
Epoch 00009: saving model to mobinetv2_unet.h5
100/100 [==============================] - 33s 328ms/step - loss: 0.0510 - precision: 0.9927 - recall: 0.9908 - acc: 0.9918 - val_loss: 0.1098 - val_precision: 0.9779 - val_recall: 0.9704 - val_acc: 0.9735
Epoch 10/20
100/100 [==============================] - ETA: 0s - loss: 0.0445 - precision: 0.9935 - recall: 0.9921 - acc: 0.9928 
Epoch 00010: saving model to mobinetv2_unet.h5
100/100 [==============================] - 33s 325ms/step - loss: 0.0445 - precision: 0.9935 - recall: 0.9921 - acc: 0.9928 - val_loss: 0.1000 - val_precision: 0.9810 - val_recall: 0.9774 - val_acc: 0.9790
Epoch 11/20
100/100 [==============================] - ETA: 0s - loss: 0.0390 - precision: 0.9943 - recall: 0.9933 - acc: 0.9938 
Epoch 00011: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 322ms/step - loss: 0.0390 - precision: 0.9943 - recall: 0.9933 - acc: 0.9938 - val_loss: 0.0765 - val_precision: 0.9833 - val_recall: 0.9809 - val_acc: 0.9819
Epoch 12/20
100/100 [==============================] - ETA: 0s - loss: 0.0327 - precision: 0.9954 - recall: 0.9947 - acc: 0.9951 
Epoch 00012: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 315ms/step - loss: 0.0327 - precision: 0.9954 - recall: 0.9947 - acc: 0.9951 - val_loss: 0.0807 - val_precision: 0.9818 - val_recall: 0.9798 - val_acc: 0.9807
Epoch 13/20
100/100 [==============================] - ETA: 0s - loss: 0.0298 - precision: 0.9955 - recall: 0.9949 - acc: 0.9952 
Epoch 00013: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 316ms/step - loss: 0.0298 - precision: 0.9955 - recall: 0.9949 - acc: 0.9952 - val_loss: 0.0845 - val_precision: 0.9821 - val_recall: 0.9805 - val_acc: 0.9811
Epoch 14/20
100/100 [==============================] - ETA: 0s - loss: 0.0268 - precision: 0.9959 - recall: 0.9954 - acc: 0.9956 
Epoch 00014: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 312ms/step - loss: 0.0268 - precision: 0.9959 - recall: 0.9954 - acc: 0.9956 - val_loss: 0.0709 - val_precision: 0.9844 - val_recall: 0.9826 - val_acc: 0.9834
Epoch 15/20
100/100 [==============================] - ETA: 0s - loss: 0.0236 - precision: 0.9964 - recall: 0.9961 - acc: 0.9962 
Epoch 00015: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 317ms/step - loss: 0.0236 - precision: 0.9964 - recall: 0.9961 - acc: 0.9962 - val_loss: 0.0721 - val_precision: 0.9844 - val_recall: 0.9832 - val_acc: 0.9837
Epoch 16/20
100/100 [==============================] - ETA: 0s - loss: 0.0237 - precision: 0.9961 - recall: 0.9958 - acc: 0.9959 
Epoch 00016: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 311ms/step - loss: 0.0237 - precision: 0.9961 - recall: 0.9958 - acc: 0.9959 - val_loss: 0.0762 - val_precision: 0.9839 - val_recall: 0.9829 - val_acc: 0.9833
Epoch 17/20
100/100 [==============================] - ETA: 0s - loss: 0.0218 - precision: 0.9961 - recall: 0.9958 - acc: 0.9959 
Epoch 00017: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 315ms/step - loss: 0.0218 - precision: 0.9961 - recall: 0.9958 - acc: 0.9959 - val_loss: 0.0765 - val_precision: 0.9821 - val_recall: 0.9811 - val_acc: 0.9816
Epoch 18/20
100/100 [==============================] - ETA: 0s - loss: 0.0193 - precision: 0.9966 - recall: 0.9963 - acc: 0.9964 
Epoch 00018: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 314ms/step - loss: 0.0193 - precision: 0.9966 - recall: 0.9963 - acc: 0.9964 - val_loss: 0.0735 - val_precision: 0.9852 - val_recall: 0.9843 - val_acc: 0.9847
Epoch 19/20
100/100 [==============================] - ETA: 0s - loss: 0.0156 - precision: 0.9973 - recall: 0.9972 - acc: 0.9973 
Epoch 00019: saving model to mobinetv2_unet.h5
100/100 [==============================] - 32s 318ms/step - loss: 0.0156 - precision: 0.9973 - recall: 0.9972 - acc: 0.9973 - val_loss: 0.0628 - val_precision: 0.9865 - val_recall: 0.9860 - val_acc: 0.9862
Epoch 20/20
100/100 [==============================] - ETA: 0s - loss: 0.0138 - precision: 0.9977 - recall: 0.9975 - acc: 0.9976 
Epoch 00020: saving model to mobinetv2_unet.h5
100/100 [==============================] - 31s 313ms/step - loss: 0.0138 - precision: 0.9977 - recall: 0.9975 - acc: 0.9976 - val_loss: 0.0706 - val_precision: 0.9849 - val_recall: 0.9844 - val_acc: 0.9846