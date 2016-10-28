#pragma once

#include "tensorNet.h"

class trailsNet : public tensorNet
{
  public:
    static trailsNet *Create(const char *prototxtPath, const char *modelPath);

    virtual ~trailsNet() = default;

    int Classify(float *rgba, uint32_t width, uint32_t height, float *confidence = NULL);

    inline const char *GetClassDesc(uint32_t index) const { return mClassDesc[index].c_str(); }

  protected:
    trailsNet();

  private:
    uint32_t mOutputClasses;
    std::vector<std::string> mClassDesc;
};