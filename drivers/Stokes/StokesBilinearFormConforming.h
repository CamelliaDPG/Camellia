#ifndef STOKES_BILINEAR_FORM_CONFORMING
#define STOKES_BILINEAR_FORM_CONFORMING

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

/*
 *  StokesBilinearFormConforming.h
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "StokesBilinearForm.h"

class StokesBilinearFormConforming : public StokesBilinearForm
{
public:
  StokesBilinearFormConforming(double mu) : StokesBilinearForm(mu) {}
  virtual EFunctionSpaceExtended functionSpaceForTrial(int trialID)
  {
    if ((trialID==StokesBilinearForm::U1_HAT)
        || (trialID==StokesBilinearForm::U2_HAT) )
    {
      return IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    }
    else
    {
      return this->StokesBilinearForm::functionSpaceForTrial(trialID);
    }
  }
};

#endif