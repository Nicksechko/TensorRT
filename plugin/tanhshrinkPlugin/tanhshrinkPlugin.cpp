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
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;

namespace
{
    const char* TANHSHRINK_PLUGIN_VERSION{"1"};
    const char* TANHSHRINK_PLUGIN_NAME{"Tanhshrink_TRT"};
} // namespace

PluginFieldCollection TanhshrinkPluginCreator::mFC{};
std::vector<PluginField> TanhshrinkPluginCreator::mPluginAttributes;

TanhshrinkPlugin::TanhshrinkPlugin() {}

TanhshrinkPlugin::TanhshrinkPlugin(nvinfer1::DataType iType, int iC, int iH, int iW, int oC, int oH, int oW)
        : iType(iType)
        , iC(iC)
        , iH(iH)
        , iW(iW)
        , oC(oC)
        , oH(oH)
        , oW(oW)
{
}

TanhshrinkPlugin::TanhshrinkPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    iC = read<int>(d);
    iH = read<int>(d);
    iW = read<int>(d);
    oC = read<int>(d);
    oH = read<int>(d);
    oW = read<int>(d);
    ASSERT(d == a + length);
}

int TanhshrinkPlugin::getNbOutputs() const
{
    return 1;
}

int TanhshrinkPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void TanhshrinkPlugin::terminate() {}

Dims TanhshrinkPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // CHW
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = inputs->nbDims;
    dimsOutput.d[0] = inputs->d[0];
    dimsOutput.d[1] = inputs->d[1];
    dimsOutput.d[2] = inputs->d[2];
    dimsOutput.d[3] = inputs->d[3];
    return dimsOutput;
}

size_t TanhshrinkPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t TanhshrinkPlugin::getSerializationSize() const
{
    // iC, iH, iW, oC, oH, oW
    return sizeof(int) * 6;
}

void TanhshrinkPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, iC);
    write(d, iH);
    write(d, iW);
    write(d, oC);
    write(d, oH);
    write(d, oW);
    ASSERT(d == a + getSerializationSize());
}

void TanhshrinkPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                                       const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
                                       const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    iC = inputDims->d[0];
    iH = inputDims->d[1];
    iW = inputDims->d[2];

    oC = outputDims->d[0];
    oH = outputDims->d[1];
    oW = outputDims->d[2];

    iType = inputTypes[0];
}

bool TanhshrinkPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return type == DataType::kFLOAT && format == PluginFormat::kNCHW;
}

const char* TanhshrinkPlugin::getPluginType() const
{
    return TANHSHRINK_PLUGIN_NAME;
}

const char* TanhshrinkPlugin::getPluginVersion() const
{
    return TANHSHRINK_PLUGIN_VERSION;
}

void TanhshrinkPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* TanhshrinkPlugin::clone() const
{
    auto* plugin = new TanhshrinkPlugin(iType, iC, iH, iW, oC, oH, oW);
    return plugin;
}

void TanhshrinkPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* TanhshrinkPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType TanhshrinkPlugin::getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool TanhshrinkPlugin::isOutputBroadcastAcrossBatch(
        int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool TanhshrinkPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
TanhshrinkPluginCreator::TanhshrinkPluginCreator() {}

const char* TanhshrinkPluginCreator::getPluginName() const
{
    return TANHSHRINK_PLUGIN_NAME;
}

const char* TanhshrinkPluginCreator::getPluginVersion() const
{
    return TANHSHRINK_PLUGIN_VERSION;
}

const PluginFieldCollection* TanhshrinkPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* TanhshrinkPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    TanhshrinkPlugin* plugin = new TanhshrinkPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* TanhshrinkPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    TanhshrinkPlugin* plugin = new TanhshrinkPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

