//
//  PhysicalCellRelativeRefinementInfo.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/15/21.
//

#ifndef PhysicalCellRelativeRefinementInfo_h
#define PhysicalCellRelativeRefinementInfo_h

namespace Camellia
{
  struct PhysicalCellRelativeRefinementInfo
  {
    GlobalIndexType physicalCellID;
    int unrefinementLevel; // relative to the ancestor that is as wide as the phase space cell
    int uniformDescendantNumber;  // in 1D, these move left to right
    double refSpaceXCoordinateInAncestor() const // left vertex coordinate; ranges from -1 to 1
    {
      int numDescendants = 1 << unrefinementLevel;
      return uniformDescendantNumber * 2.0 / numDescendants - 1.0;
    }
  };
}

#endif /* PhysicalCellRelativeRefinementInfo_h */
