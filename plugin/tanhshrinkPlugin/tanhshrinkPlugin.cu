/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "tanhshrinkPlugin.h"
#include <cuda_fp16.h>
#include <cmath>

template <typename T_DATA>
     __global__ void kernelTanhshrink(
         T_DATA* inputs,
         T_DATA* outputs,
         int N
         ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
         outputs[index] = inputs[index] - tanh(inputs[index]);
    }
}

 template <typename T>
 int inferenceTanhshrink(
     int batchSize,
     int iC,
     int iH,
     int iW,
     int oC,
     int oH,
     int oW,
     T* inputs,
     T* outputs,
     cudaStream_t stream){
     // NCHW
     const int nThreads = 512;
     int size = batchSize * iC * iH * iW;

     int nBlocks = (int)((float) size / nThreads) + 1;

     kernelTanhshrink <<< nBlocks, nThreads, 0, stream >>> (inputs, outputs, size);
     cudaDeviceSynchronize();

     cudaError_t err = cudaGetLastError();
     if ( cudaSuccess != err )
     {
         fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
         return 1;
     }
     return 0;
 }

 int TanhshrinkPlugin::enqueue(
     int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
 {
     switch(iType){
         case DataType::kFLOAT:
             return inferenceTanhshrink(batchSize, iC, iH, iW, oC, oH, oW, (float*)inputs[0], (float*)outputs[0], stream);
     }
     return 1;
 }
