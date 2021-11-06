(tf2.3) quannm@quannm:/media/HDD/bkai$ python main.py
2021-11-06 15:41:38.593741: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
Amount of images: 1000
Training: 800 - Validation: 200
100 - 26
2021-11-06 15:41:39.792765: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2021-11-06 15:41:39.819363: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:39.819504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 15:41:39.819521: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:41:39.820871: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 15:41:39.822085: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 15:41:39.822276: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 15:41:39.823550: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 15:41:39.824221: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 15:41:39.826955: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:41:39.827083: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:39.827281: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:39.827386: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 15:41:39.827642: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-06 15:41:39.832761: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3199980000 Hz
2021-11-06 15:41:39.833179: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56222e771ff0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-11-06 15:41:39.833199: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-11-06 15:41:39.899602: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:39.899767: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5622306d6e10 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-11-06 15:41:39.899781: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): NVIDIA GeForce GTX 1070 Ti, Compute Capability 6.1
2021-11-06 15:41:39.899959: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:39.900079: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 7.92GiB deviceMemoryBandwidth: 238.66GiB/s
2021-11-06 15:41:39.900106: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:41:39.900164: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
2021-11-06 15:41:39.900191: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcufft.so.10
2021-11-06 15:41:39.900208: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcurand.so.10
2021-11-06 15:41:39.900233: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusolver.so.10
2021-11-06 15:41:39.900260: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcusparse.so.10
2021-11-06 15:41:39.900286: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:41:39.900363: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:39.900514: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:39.900617: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1858] Adding visible gpu devices: 0
2021-11-06 15:41:39.900644: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudart.so.10.1
2021-11-06 15:41:40.178688: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-11-06 15:41:40.178717: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      0 
2021-11-06 15:41:40.178732: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1276] 0:   N 
2021-11-06 15:41:40.178918: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:40.179273: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:982] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-11-06 15:41:40.179412: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1402] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5982 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1070 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Using unet model 
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
concatenate (Concatenate)       (None, 32, 32, 1024) 0           conv2d_transpose[0][0]           
                                                                 activation_7[0][0]               
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 512)  4719104     concatenate[0][0]                
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 32, 32, 512)  2048        conv2d_10[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 32, 32, 512)  0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 32, 32, 512)  2359808     activation_10[0][0]              
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 32, 32, 512)  2048        conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 32, 32, 512)  0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 256)  524544      activation_11[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 64, 64, 512)  0           conv2d_transpose_1[0][0]         
                                                                 activation_5[0][0]               
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 64, 64, 256)  1179904     concatenate_1[0][0]              
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 64, 64, 256)  1024        conv2d_12[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 64, 64, 256)  0           batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 64, 64, 256)  590080      activation_12[0][0]              
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 64, 64, 256)  1024        conv2d_13[0][0]                  
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 64, 64, 256)  0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 128, 128, 128 131200      activation_13[0][0]              
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 128, 128, 256 0           conv2d_transpose_2[0][0]         
                                                                 activation_3[0][0]               
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 128, 128, 128 295040      concatenate_2[0][0]              
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 128, 128, 128 512         conv2d_14[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 128, 128, 128 0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 128, 128, 128 147584      activation_14[0][0]              
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 128, 128, 128 512         conv2d_15[0][0]                  
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 128, 128, 128 0           batch_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTrans (None, 256, 256, 64) 32832       activation_15[0][0]              
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 256, 256, 128 0           conv2d_transpose_3[0][0]         
                                                                 activation_1[0][0]               
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 256, 256, 64) 73792       concatenate_3[0][0]              
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 256, 256, 64) 256         conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 256, 256, 64) 0           batch_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 256, 256, 64) 36928       activation_16[0][0]              
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 256, 256, 64) 256         conv2d_17[0][0]                  
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 256, 256, 64) 0           batch_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 256, 256, 3)  195         activation_17[0][0]              
==================================================================================================
Total params: 31,055,427
Trainable params: 31,043,651
Non-trainable params: 11,776
__________________________________________________________________________________________________
Epoch 1/20
2021-11-06 15:41:42.388957: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcudnn.so.7
2021-11-06 15:41:42.906110: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcublas.so.10
100/100 [==============================] - ETA: 0s - loss: 0.3680 - precision: 0.9397 - recall: 0.8881 - acc: 0.9250 
Epoch 00001: saving model to unet.h5
100/100 [==============================] - 45s 445ms/step - loss: 0.3680 - precision: 0.9397 - recall: 0.8881 - acc: 0.9250 - val_loss: 0.4591 - val_precision: 0.9484 - val_recall: 0.9482 - val_acc: 0.9485
Epoch 2/20
100/100 [==============================] - ETA: 0s - loss: 0.2220 - precision: 0.9510 - recall: 0.9461 - acc: 0.9488 
Epoch 00002: saving model to unet.h5
100/100 [==============================] - 45s 448ms/step - loss: 0.2220 - precision: 0.9510 - recall: 0.9461 - acc: 0.9488 - val_loss: 0.2821 - val_precision: 0.9489 - val_recall: 0.9489 - val_acc: 0.9489
Epoch 3/20
100/100 [==============================] - ETA: 0s - loss: 0.1940 - precision: 0.9528 - recall: 0.9475 - acc: 0.9503 
Epoch 00003: saving model to unet.h5
100/100 [==============================] - 44s 444ms/step - loss: 0.1940 - precision: 0.9528 - recall: 0.9475 - acc: 0.9503 - val_loss: 0.2419 - val_precision: 0.9484 - val_recall: 0.9478 - val_acc: 0.9482
Epoch 4/20
100/100 [==============================] - ETA: 0s - loss: 0.1752 - precision: 0.9555 - recall: 0.9511 - acc: 0.9534 
Epoch 00004: saving model to unet.h5
100/100 [==============================] - 43s 431ms/step - loss: 0.1752 - precision: 0.9555 - recall: 0.9511 - acc: 0.9534 - val_loss: 0.2112 - val_precision: 0.9493 - val_recall: 0.9461 - val_acc: 0.9478
Epoch 5/20
100/100 [==============================] - ETA: 0s - loss: 0.1640 - precision: 0.9575 - recall: 0.9536 - acc: 0.9556 
Epoch 00005: saving model to unet.h5
100/100 [==============================] - 43s 431ms/step - loss: 0.1640 - precision: 0.9575 - recall: 0.9536 - acc: 0.9556 - val_loss: 0.2559 - val_precision: 0.9139 - val_recall: 0.9020 - val_acc: 0.9086
Epoch 6/20
100/100 [==============================] - ETA: 0s - loss: 0.1482 - precision: 0.9615 - recall: 0.9577 - acc: 0.9597 
Epoch 00006: saving model to unet.h5
100/100 [==============================] - 44s 440ms/step - loss: 0.1482 - precision: 0.9615 - recall: 0.9577 - acc: 0.9597 - val_loss: 0.1568 - val_precision: 0.9547 - val_recall: 0.9499 - val_acc: 0.9524
Epoch 7/20
100/100 [==============================] - ETA: 0s - loss: 0.1400 - precision: 0.9630 - recall: 0.9597 - acc: 0.9614 
Epoch 00007: saving model to unet.h5
100/100 [==============================] - 44s 439ms/step - loss: 0.1400 - precision: 0.9630 - recall: 0.9597 - acc: 0.9614 - val_loss: 0.1339 - val_precision: 0.9630 - val_recall: 0.9598 - val_acc: 0.9614
Epoch 8/20
100/100 [==============================] - ETA: 0s - loss: 0.1273 - precision: 0.9668 - recall: 0.9631 - acc: 0.9650 
Epoch 00008: saving model to unet.h5
100/100 [==============================] - 43s 434ms/step - loss: 0.1273 - precision: 0.9668 - recall: 0.9631 - acc: 0.9650 - val_loss: 0.1160 - val_precision: 0.9677 - val_recall: 0.9652 - val_acc: 0.9665
Epoch 9/20
100/100 [==============================] - ETA: 0s - loss: 0.1147 - precision: 0.9701 - recall: 0.9664 - acc: 0.9682 
Epoch 00009: saving model to unet.h5
100/100 [==============================] - 43s 433ms/step - loss: 0.1147 - precision: 0.9701 - recall: 0.9664 - acc: 0.9682 - val_loss: 0.1009 - val_precision: 0.9731 - val_recall: 0.9708 - val_acc: 0.9719
Epoch 10/20
100/100 [==============================] - ETA: 0s - loss: 0.1061 - precision: 0.9726 - recall: 0.9687 - acc: 0.9706 
Epoch 00010: saving model to unet.h5
100/100 [==============================] - 43s 426ms/step - loss: 0.1061 - precision: 0.9726 - recall: 0.9687 - acc: 0.9706 - val_loss: 0.0997 - val_precision: 0.9736 - val_recall: 0.9708 - val_acc: 0.9722
Epoch 11/20
100/100 [==============================] - ETA: 0s - loss: 0.0947 - precision: 0.9761 - recall: 0.9721 - acc: 0.9740 
Epoch 00011: saving model to unet.h5
100/100 [==============================] - 44s 440ms/step - loss: 0.0947 - precision: 0.9761 - recall: 0.9721 - acc: 0.9740 - val_loss: 0.0986 - val_precision: 0.9734 - val_recall: 0.9698 - val_acc: 0.9716
Epoch 12/20
100/100 [==============================] - ETA: 0s - loss: 0.0925 - precision: 0.9762 - recall: 0.9716 - acc: 0.9738 
Epoch 00012: saving model to unet.h5
100/100 [==============================] - 44s 437ms/step - loss: 0.0925 - precision: 0.9762 - recall: 0.9716 - acc: 0.9738 - val_loss: 0.0861 - val_precision: 0.9768 - val_recall: 0.9728 - val_acc: 0.9748
Epoch 13/20
100/100 [==============================] - ETA: 0s - loss: 0.0819 - precision: 0.9794 - recall: 0.9751 - acc: 0.9771 
Epoch 00013: saving model to unet.h5
100/100 [==============================] - 43s 435ms/step - loss: 0.0819 - precision: 0.9794 - recall: 0.9751 - acc: 0.9771 - val_loss: 0.0886 - val_precision: 0.9763 - val_recall: 0.9727 - val_acc: 0.9746
Epoch 14/20
100/100 [==============================] - ETA: 0s - loss: 0.0789 - precision: 0.9798 - recall: 0.9758 - acc: 0.9777 
Epoch 00014: saving model to unet.h5
100/100 [==============================] - 43s 431ms/step - loss: 0.0789 - precision: 0.9798 - recall: 0.9758 - acc: 0.9777 - val_loss: 0.1073 - val_precision: 0.9690 - val_recall: 0.9633 - val_acc: 0.9662
Epoch 15/20
100/100 [==============================] - ETA: 0s - loss: 0.0722 - precision: 0.9820 - recall: 0.9772 - acc: 0.9795 
Epoch 00015: saving model to unet.h5
100/100 [==============================] - 44s 437ms/step - loss: 0.0722 - precision: 0.9820 - recall: 0.9772 - acc: 0.9795 - val_loss: 0.0752 - val_precision: 0.9810 - val_recall: 0.9772 - val_acc: 0.9790
Epoch 16/20
100/100 [==============================] - ETA: 0s - loss: 0.0605 - precision: 0.9852 - recall: 0.9809 - acc: 0.9828 
Epoch 00016: saving model to unet.h5
100/100 [==============================] - 43s 432ms/step - loss: 0.0605 - precision: 0.9852 - recall: 0.9809 - acc: 0.9828 - val_loss: 0.0881 - val_precision: 0.9777 - val_recall: 0.9739 - val_acc: 0.9759
Epoch 17/20
100/100 [==============================] - ETA: 0s - loss: 0.0629 - precision: 0.9842 - recall: 0.9797 - acc: 0.9818 
Epoch 00017: saving model to unet.h5
100/100 [==============================] - 43s 433ms/step - loss: 0.0629 - precision: 0.9842 - recall: 0.9797 - acc: 0.9818 - val_loss: 0.0791 - val_precision: 0.9782 - val_recall: 0.9754 - val_acc: 0.9767
Epoch 18/20
100/100 [==============================] - ETA: 0s - loss: 0.0593 - precision: 0.9849 - recall: 0.9803 - acc: 0.9824 
Epoch 00018: saving model to unet.h5
100/100 [==============================] - 43s 429ms/step - loss: 0.0593 - precision: 0.9849 - recall: 0.9803 - acc: 0.9824 - val_loss: 0.1049 - val_precision: 0.9684 - val_recall: 0.9641 - val_acc: 0.9661
Epoch 19/20
100/100 [==============================] - ETA: 0s - loss: 0.0542 - precision: 0.9859 - recall: 0.9814 - acc: 0.9834 
Epoch 00019: saving model to unet.h5
100/100 [==============================] - 43s 434ms/step - loss: 0.0542 - precision: 0.9859 - recall: 0.9814 - acc: 0.9834 - val_loss: 0.0741 - val_precision: 0.9812 - val_recall: 0.9763 - val_acc: 0.9785
Epoch 20/20
100/100 [==============================] - ETA: 0s - loss: 0.0444 - precision: 0.9891 - recall: 0.9850 - acc: 0.9869 
Epoch 00020: saving model to unet.h5
100/100 [==============================] - 44s 443ms/step - loss: 0.0444 - precision: 0.9891 - recall: 0.9850 - acc: 0.9869 - val_loss: 0.0732 - val_precision: 0.9810 - val_recall: 0.9769 - val_acc: 0.9787