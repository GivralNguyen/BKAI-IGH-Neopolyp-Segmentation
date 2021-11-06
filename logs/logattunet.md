(tf2.3) quannm@quannm:/media/HDD/bkai$ python main.py
2021-11-06 15:58:17.408887: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Amount of images: 1000
Training: 800 - Validation: 200
100 - 26
2021-11-06 15:58:18.883269: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2021-11-06 15:58:18.911339: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:18.911635: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 15:58:18.911655: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:58:18.920569: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 15:58:18.925176: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 15:58:18.930584: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 15:58:18.933558: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 15:58:18.935026: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 15:58:18.940736: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:58:18.940875: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:18.941069: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:18.941185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 15:58:18.941499: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-06 15:58:18.946216: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3199980000 Hz
2021-11-06 15:58:18.946607: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5598168c1ff0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-11-06 15:58:18.946630: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-11-06 15:58:19.005592: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:19.005804: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559818783c50 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-11-06 15:58:19.005821: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce GTX 1070 Ti, Compute Capability 6.1
2021-11-06 15:58:19.005994: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:19.006129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 15:58:19.006161: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:58:19.006211: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 15:58:19.006236: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 15:58:19.006252: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 15:58:19.006269: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 15:58:19.006290: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 15:58:19.006311: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:58:19.006381: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:19.006604: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:19.006737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 15:58:19.006768: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:58:19.291947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-06 15:58:19.291975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2021-11-06 15:58:19.291982: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2021-11-06 15:58:19.292137: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:19.292307: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:58:19.292424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5946 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Using att_unet model 
Model: "U-Net"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 256, 256, 3) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 256, 256, 64) 1792        input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 256, 256, 64) 256         conv2d[0][0]                     
__________________________________________________________________________________________________
activation (Activation)         (None, 256, 256, 64) 0           batch_normalization[0][0]        
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 256, 256, 64) 36928       activation[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 256, 256, 64) 256         conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 256, 256, 64) 0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 128, 128, 64) 0           activation_1[0][0]               
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 128, 128, 128 73856       max_pooling2d[0][0]              
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 128, 128, 128 512         conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 128, 128, 128 0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 128, 128, 128 147584      activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 128, 128, 128 512         conv2d_3[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 128, 128, 128 0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 128)  0           activation_3[0][0]               
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 64, 64, 256)  295168      max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 64, 64, 256)  1024        conv2d_4[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 64, 64, 256)  0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 64, 64, 256)  590080      activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 64, 64, 256)  1024        conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 64, 64, 256)  0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 256)  0           activation_5[0][0]               
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 512)  1180160     max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 512)  2048        conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 512)  0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 512)  2359808     activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 512)  2048        conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 512)  0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 512)  0           activation_7[0][0]               
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 16, 16, 1024) 4719616     max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 16, 16, 1024) 4096        conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 16, 16, 1024) 0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 16, 16, 1024) 9438208     activation_8[0][0]               
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 16, 16, 1024) 4096        conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 16, 16, 1024) 0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_transpose (Conv2DTranspo (None, 32, 32, 512)  2097664     activation_9[0][0]               
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 256)  131328      conv2d_transpose[0][0]           
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 256)  1179904     activation_7[0][0]               
__________________________________________________________________________________________________
add (Add)                       (None, 32, 32, 256)  0           conv2d_10[0][0]                  
                                                                 conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 32, 32, 256)  0           add[0][0]                        
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 1)    257         activation_10[0][0]              
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 32, 32, 1)    0           conv2d_12[0][0]                  
__________________________________________________________________________________________________
up_sampling2d (UpSampling2D)    (None, 32, 32, 1)    0           activation_11[0][0]              
__________________________________________________________________________________________________
lambda (Lambda)                 (None, 32, 32, 512)  0           up_sampling2d[0][0]              
__________________________________________________________________________________________________
multiply (Multiply)             (None, 32, 32, 512)  0           lambda[0][0]                     
                                                                 activation_7[0][0]               
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 32, 32, 512)  262656      multiply[0][0]                   
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 32, 32, 512)  2048        conv2d_13[0][0]                  
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 32, 32, 1024) 0           conv2d_transpose[0][0]           
                                                                 batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 32, 32, 512)  4719104     concatenate[0][0]                
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 32, 32, 512)  2048        conv2d_14[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 32, 32, 512)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 32, 32, 512)  2359808     activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 32, 32, 512)  2048        conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 32, 32, 512)  0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 256)  524544      activation_13[0][0]              
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 64, 64, 128)  32896       conv2d_transpose_1[0][0]         
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 64, 64, 128)  295040      activation_5[0][0]               
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 64, 128)  0           conv2d_16[0][0]                  
                                                                 conv2d_17[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 64, 64, 128)  0           add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 64, 64, 1)    129         activation_14[0][0]              
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 64, 64, 1)    0           conv2d_18[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 64, 64, 1)    0           activation_15[0][0]              
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 64, 64, 256)  0           up_sampling2d_1[0][0]            
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 64, 64, 256)  0           lambda_1[0][0]                   
                                                                 activation_5[0][0]               
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 64, 64, 256)  65792       multiply_1[0][0]                 
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 64, 64, 256)  1024        conv2d_19[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 64, 64, 512)  0           conv2d_transpose_1[0][0]         
                                                                 batch_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 64, 64, 256)  1179904     concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 64, 64, 256)  1024        conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 64, 64, 256)  0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 64, 64, 256)  590080      activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 64, 64, 256)  1024        conv2d_21[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 64, 64, 256)  0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 128, 128, 128 131200      activation_17[0][0]              
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 128, 128, 64) 8256        conv2d_transpose_2[0][0]         
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 128, 128, 64) 73792       activation_3[0][0]               
__________________________________________________________________________________________________
add_2 (Add)                     (None, 128, 128, 64) 0           conv2d_22[0][0]                  
                                                                 conv2d_23[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 128, 128, 64) 0           add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 128, 128, 1)  65          activation_18[0][0]              
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 128, 128, 1)  0           conv2d_24[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)  (None, 128, 128, 1)  0           activation_19[0][0]              
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 128, 128, 128 0           up_sampling2d_2[0][0]            
__________________________________________________________________________________________________
multiply_2 (Multiply)           (None, 128, 128, 128 0           lambda_2[0][0]                   
                                                                 activation_3[0][0]               
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 128, 128, 128 16512       multiply_2[0][0]                 
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 128, 128, 128 512         conv2d_25[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 128, 128, 256 0           conv2d_transpose_2[0][0]         
                                                                 batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 128, 128, 128 295040      concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 128, 128, 128 512         conv2d_26[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 128, 128, 128 0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 128, 128, 128 147584      activation_20[0][0]              
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 128, 128, 128 512         conv2d_27[0][0]                  
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 128, 128, 128 0           batch_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 256, 256, 64) 32832       activation_21[0][0]              
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 256, 256, 32) 2080        conv2d_transpose_3[0][0]         
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 256, 256, 32) 18464       activation_1[0][0]               
__________________________________________________________________________________________________
add_3 (Add)                     (None, 256, 256, 32) 0           conv2d_28[0][0]                  
                                                                 conv2d_29[0][0]                  
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 256, 256, 32) 0           add_3[0][0]                      
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 256, 256, 1)  33          activation_22[0][0]              
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 256, 256, 1)  0           conv2d_30[0][0]                  
__________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)  (None, 256, 256, 1)  0           activation_23[0][0]              
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, 256, 256, 64) 0           up_sampling2d_3[0][0]            
__________________________________________________________________________________________________
multiply_3 (Multiply)           (None, 256, 256, 64) 0           lambda_3[0][0]                   
                                                                 activation_1[0][0]               
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 256, 256, 64) 4160        multiply_3[0][0]                 
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 256, 256, 64) 256         conv2d_31[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256, 256, 128 0           conv2d_transpose_3[0][0]         
                                                                 batch_normalization_19[0][0]     
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 256, 256, 64) 73792       concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 256, 256, 64) 256         conv2d_32[0][0]                  
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 256, 256, 64) 0           batch_normalization_20[0][0]     
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 256, 256, 64) 36928       activation_24[0][0]              
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 256, 256, 64) 256         conv2d_33[0][0]                  
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 256, 256, 64) 0           batch_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 256, 256, 3)  195         activation_25[0][0]              
==================================================================================================
Total params: 33,150,631
Trainable params: 33,136,935
Non-trainable params: 13,696
__________________________________________________________________________________________________
Epoch 1/40
2021-11-06 15:58:22.778982: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:58:23.303305: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 15:58:24.802601: W tensorflow/core/common_runtime/bfc_allocator.cc:246] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.27GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-06 15:58:25.098693: W tensorflow/core/common_runtime/bfc_allocator.cc:246] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.70GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-06 15:58:25.230725: W tensorflow/core/common_runtime/bfc_allocator.cc:246] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.16GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2021-11-06 15:58:25.250601: W tensorflow/core/common_runtime/bfc_allocator.cc:246] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.28GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
  2/100 [..............................] - ETA: 23s - loss: 0.8009 - precision: 0.7532 - recall: 0.5354 - acc: 0.7230WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.1718s vs `on_train_batch_end` time: 0.3166s). Check your callbacks.
100/100 [==============================] - ETA: 0s - loss: 0.3335 - precision: 0.9395 - recall: 0.9100 - acc: 0.9300 
Epoch 00001: saving model to att_unet.h5
100/100 [==============================] - 56s 564ms/step - loss: 0.3335 - precision: 0.9395 - recall: 0.9100 - acc: 0.9300 - val_loss: 0.5351 - val_precision: 0.9468 - val_recall: 0.9442 - val_acc: 0.9458
Epoch 2/40
100/100 [==============================] - ETA: 0s - loss: 0.1967 - precision: 0.9530 - recall: 0.9476 - acc: 0.9506 
Epoch 00002: saving model to att_unet.h5
100/100 [==============================] - 57s 569ms/step - loss: 0.1967 - precision: 0.9530 - recall: 0.9476 - acc: 0.9506 - val_loss: 0.7507 - val_precision: 0.5465 - val_recall: 0.4623 - val_acc: 0.5351
Epoch 3/40
100/100 [==============================] - ETA: 0s - loss: 0.1743 - precision: 0.9556 - recall: 0.9502 - acc: 0.9530 
Epoch 00003: saving model to att_unet.h5
100/100 [==============================] - 56s 562ms/step - loss: 0.1743 - precision: 0.9556 - recall: 0.9502 - acc: 0.9530 - val_loss: 0.2700 - val_precision: 0.9507 - val_recall: 0.9414 - val_acc: 0.9466
Epoch 4/40
100/100 [==============================] - ETA: 0s - loss: 0.1562 - precision: 0.9588 - recall: 0.9543 - acc: 0.9566 
Epoch 00004: saving model to att_unet.h5
100/100 [==============================] - 56s 559ms/step - loss: 0.1562 - precision: 0.9588 - recall: 0.9543 - acc: 0.9566 - val_loss: 0.2989 - val_precision: 0.9104 - val_recall: 0.8816 - val_acc: 0.8976
Epoch 5/40
100/100 [==============================] - ETA: 0s - loss: 0.1421 - precision: 0.9617 - recall: 0.9575 - acc: 0.9596 
Epoch 00005: saving model to att_unet.h5
100/100 [==============================] - 56s 563ms/step - loss: 0.1421 - precision: 0.9617 - recall: 0.9575 - acc: 0.9596 - val_loss: 0.1563 - val_precision: 0.9588 - val_recall: 0.9552 - val_acc: 0.9571
Epoch 6/40
100/100 [==============================] - ETA: 0s - loss: 0.1309 - precision: 0.9651 - recall: 0.9613 - acc: 0.9632 
Epoch 00006: saving model to att_unet.h5
100/100 [==============================] - 56s 557ms/step - loss: 0.1309 - precision: 0.9651 - recall: 0.9613 - acc: 0.9632 - val_loss: 0.1209 - val_precision: 0.9705 - val_recall: 0.9636 - val_acc: 0.9674
Epoch 7/40
100/100 [==============================] - ETA: 0s - loss: 0.1198 - precision: 0.9685 - recall: 0.9642 - acc: 0.9664 
Epoch 00007: saving model to att_unet.h5
100/100 [==============================] - 55s 555ms/step - loss: 0.1198 - precision: 0.9685 - recall: 0.9642 - acc: 0.9664 - val_loss: 0.1061 - val_precision: 0.9732 - val_recall: 0.9701 - val_acc: 0.9716
Epoch 8/40
100/100 [==============================] - ETA: 0s - loss: 0.1092 - precision: 0.9715 - recall: 0.9676 - acc: 0.9696 
Epoch 00008: saving model to att_unet.h5
100/100 [==============================] - 55s 546ms/step - loss: 0.1092 - precision: 0.9715 - recall: 0.9676 - acc: 0.9696 - val_loss: 0.1071 - val_precision: 0.9716 - val_recall: 0.9680 - val_acc: 0.9698
Epoch 9/40
100/100 [==============================] - ETA: 0s - loss: 0.0990 - precision: 0.9743 - recall: 0.9705 - acc: 0.9724 
Epoch 00009: saving model to att_unet.h5
100/100 [==============================] - 55s 547ms/step - loss: 0.0990 - precision: 0.9743 - recall: 0.9705 - acc: 0.9724 - val_loss: 0.1050 - val_precision: 0.9759 - val_recall: 0.9656 - val_acc: 0.9710
Epoch 10/40
100/100 [==============================] - ETA: 0s - loss: 0.0924 - precision: 0.9765 - recall: 0.9723 - acc: 0.9743 
Epoch 00010: saving model to att_unet.h5
100/100 [==============================] - 56s 557ms/step - loss: 0.0924 - precision: 0.9765 - recall: 0.9723 - acc: 0.9743 - val_loss: 0.0928 - val_precision: 0.9742 - val_recall: 0.9709 - val_acc: 0.9725
Epoch 11/40
100/100 [==============================] - ETA: 0s - loss: 0.0831 - precision: 0.9789 - recall: 0.9749 - acc: 0.9768 
Epoch 00011: saving model to att_unet.h5
100/100 [==============================] - 55s 553ms/step - loss: 0.0831 - precision: 0.9789 - recall: 0.9749 - acc: 0.9768 - val_loss: 0.1150 - val_precision: 0.9724 - val_recall: 0.9608 - val_acc: 0.9668
Epoch 12/40
100/100 [==============================] - ETA: 0s - loss: 0.0797 - precision: 0.9796 - recall: 0.9749 - acc: 0.9770 
Epoch 00012: saving model to att_unet.h5
100/100 [==============================] - 57s 572ms/step - loss: 0.0797 - precision: 0.9796 - recall: 0.9749 - acc: 0.9770 - val_loss: 0.0968 - val_precision: 0.9760 - val_recall: 0.9729 - val_acc: 0.9743
Epoch 13/40
100/100 [==============================] - ETA: 0s - loss: 0.0695 - precision: 0.9826 - recall: 0.9785 - acc: 0.9804 
Epoch 00013: saving model to att_unet.h5
100/100 [==============================] - 57s 571ms/step - loss: 0.0695 - precision: 0.9826 - recall: 0.9785 - acc: 0.9804 - val_loss: 0.0824 - val_precision: 0.9800 - val_recall: 0.9747 - val_acc: 0.9773
Epoch 14/40
100/100 [==============================] - ETA: 0s - loss: 0.0704 - precision: 0.9816 - recall: 0.9770 - acc: 0.9791 
Epoch 00014: saving model to att_unet.h5
100/100 [==============================] - 57s 566ms/step - loss: 0.0704 - precision: 0.9816 - recall: 0.9770 - acc: 0.9791 - val_loss: 0.0816 - val_precision: 0.9789 - val_recall: 0.9761 - val_acc: 0.9774
Epoch 15/40
100/100 [==============================] - ETA: 0s - loss: 0.0608 - precision: 0.9845 - recall: 0.9797 - acc: 0.9819 
Epoch 00015: saving model to att_unet.h5
100/100 [==============================] - 57s 566ms/step - loss: 0.0608 - precision: 0.9845 - recall: 0.9797 - acc: 0.9819 - val_loss: 0.0808 - val_precision: 0.9803 - val_recall: 0.9738 - val_acc: 0.9766
Epoch 16/40
100/100 [==============================] - ETA: 0s - loss: 0.0553 - precision: 0.9862 - recall: 0.9817 - acc: 0.9838 
Epoch 00016: saving model to att_unet.h5
100/100 [==============================] - 56s 564ms/step - loss: 0.0553 - precision: 0.9862 - recall: 0.9817 - acc: 0.9838 - val_loss: 0.0774 - val_precision: 0.9799 - val_recall: 0.9757 - val_acc: 0.9778
Epoch 17/40
100/100 [==============================] - ETA: 0s - loss: 0.0542 - precision: 0.9861 - recall: 0.9813 - acc: 0.9836 
Epoch 00017: saving model to att_unet.h5
100/100 [==============================] - 58s 581ms/step - loss: 0.0542 - precision: 0.9861 - recall: 0.9813 - acc: 0.9836 - val_loss: 0.0746 - val_precision: 0.9807 - val_recall: 0.9774 - val_acc: 0.9789
Epoch 18/40
100/100 [==============================] - ETA: 0s - loss: 0.0472 - precision: 0.9877 - recall: 0.9842 - acc: 0.9858 
Epoch 00018: saving model to att_unet.h5
100/100 [==============================] - 60s 595ms/step - loss: 0.0472 - precision: 0.9877 - recall: 0.9842 - acc: 0.9858 - val_loss: 0.0880 - val_precision: 0.9790 - val_recall: 0.9756 - val_acc: 0.9772
Epoch 19/40
100/100 [==============================] - ETA: 0s - loss: 0.0422 - precision: 0.9886 - recall: 0.9856 - acc: 0.9869 
Epoch 00019: saving model to att_unet.h5
100/100 [==============================] - 55s 553ms/step - loss: 0.0422 - precision: 0.9886 - recall: 0.9856 - acc: 0.9869 - val_loss: 0.0699 - val_precision: 0.9812 - val_recall: 0.9783 - val_acc: 0.9797
Epoch 20/40
100/100 [==============================] - ETA: 0s - loss: 0.0425 - precision: 0.9887 - recall: 0.9851 - acc: 0.9867 
Epoch 00020: saving model to att_unet.h5
100/100 [==============================] - 55s 551ms/step - loss: 0.0425 - precision: 0.9887 - recall: 0.9851 - acc: 0.9867 - val_loss: 0.0772 - val_precision: 0.9800 - val_recall: 0.9765 - val_acc: 0.9781
Epoch 21/40
100/100 [==============================] - ETA: 0s - loss: 0.0398 - precision: 0.9890 - recall: 0.9861 - acc: 0.9874 
Epoch 00021: saving model to att_unet.h5
100/100 [==============================] - 57s 568ms/step - loss: 0.0398 - precision: 0.9890 - recall: 0.9861 - acc: 0.9874 - val_loss: 0.0717 - val_precision: 0.9818 - val_recall: 0.9779 - val_acc: 0.9795
Epoch 22/40
100/100 [==============================] - ETA: 0s - loss: 0.0443 - precision: 0.9877 - recall: 0.9845 - acc: 0.9860 
Epoch 00022: saving model to att_unet.h5
100/100 [==============================] - 57s 572ms/step - loss: 0.0443 - precision: 0.9877 - recall: 0.9845 - acc: 0.9860 - val_loss: 0.0915 - val_precision: 0.9785 - val_recall: 0.9754 - val_acc: 0.9768
Epoch 23/40
100/100 [==============================] - ETA: 0s - loss: 0.0338 - precision: 0.9905 - recall: 0.9885 - acc: 0.9894 
Epoch 00023: saving model to att_unet.h5
100/100 [==============================] - 56s 556ms/step - loss: 0.0338 - precision: 0.9905 - recall: 0.9885 - acc: 0.9894 - val_loss: 0.0858 - val_precision: 0.9791 - val_recall: 0.9763 - val_acc: 0.9774
Epoch 24/40
100/100 [==============================] - ETA: 0s - loss: 0.0322 - precision: 0.9909 - recall: 0.9892 - acc: 0.9900 
Epoch 00024: saving model to att_unet.h5

Epoch 00024: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-06.
100/100 [==============================] - 56s 558ms/step - loss: 0.0322 - precision: 0.9909 - recall: 0.9892 - acc: 0.9900 - val_loss: 0.0806 - val_precision: 0.9802 - val_recall: 0.9779 - val_acc: 0.9789
Epoch 00024: early stopping