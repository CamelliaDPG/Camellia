//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

/*@HEADER
 // ***********************************************************************
 //
 //                  Camellia Cubature Factory:
 //
 //  A wrapper around Intrepid's DefaultCubatureFactory that adds support
 //  for Camellia's tensor-product topologies.
 //
 // ***********************************************************************
 //@HEADER
 */

#include "CubatureFactory.h"

#include "Intrepid2_CubatureDirectLineGauss.hpp"

using namespace Teuchos;

using namespace Camellia;

Teuchos::RCP< Intrepid2::Cubature<DeviceType,double,double> > CubatureFactory::create(CellTopoPtr cellTopo, int cubDegree)
{
  Teuchos::RCP< Cubature > shardsTopoCub;
  if (cellTopo->getShardsTopology().getDimension() != 0)
  {
    shardsTopoCub = Intrepid2::DefaultCubatureFactory::create<DeviceType>(cellTopo->getShardsTopology(), cubDegree);
  }
  else
  {
    // base topology is a Node (dimension 0)
    if (cellTopo->getTensorialDegree() == 0)
    {
      return Teuchos::null;
    }
    else
    {
      CellTopoPtr line = CellTopology::line();
      CellTopoPtr equivalentTopo = CellTopology::cellTopology(line, cellTopo->getTensorialDegree()-1);
      return create(equivalentTopo, cubDegree);
    }
  }

  if (cellTopo->getTensorialDegree() == 0)
  {
    return shardsTopoCub;
  }
  else
  {
    shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );

    using CubatureDirectLine = Intrepid2::CubatureDirectLineGauss<DeviceType,double,double>;
    CubatureDirectLine lineCub(cubDegree);

    using CubatureTensor = Intrepid2::CubatureTensor<DeviceType,double,double>;
    
    Teuchos::RCP<CubatureTensor> cubatureTensor;
    
    // we want to create the first tensor product from shardsTopoCub and lineCub
    // it's a little complicated because of the way that CubatureTensor is constructed
    // shardsTopoCub is either some sort of CubatureDirect object, or it is itself a CubatureTensor
    CubatureTensor* shardsTopoCubAsTensor = dynamic_cast<CubatureTensor*>(shardsTopoCub.get());
    if (shardsTopoCubAsTensor)
    {
      cubatureTensor = Teuchos::rcp(new CubatureTensor(*shardsTopoCubAsTensor, lineCub));
    }
    else
    {
      using CubatureDirect = Intrepid2::CubatureDirect<DeviceType,double,double>;
      CubatureDirect* cubatureDirectPtr = dynamic_cast<CubatureDirect*>(shardsTopoCub.get());
      
      TEUCHOS_TEST_FOR_EXCEPTION(cubatureDirectPtr == NULL, std::invalid_argument, "shardsTopoCub is not a subclass of CubatureDirect or CubatureTensor");
      
      CubatureDirect cubatureDirect = *cubatureDirectPtr;
      
      cubatureTensor = Teuchos::rcp(new CubatureTensor(*cubatureDirectPtr, lineCub));
    }
    
    for (int tensorialComponent=1; tensorialComponent<cellTopo->getTensorialDegree(); tensorialComponent++)
    {
      cubatureTensor = Teuchos::rcp( new CubatureTensor(*cubatureTensor, lineCub) );
    }
    return cubatureTensor;
  }
}

Teuchos::RCP< Intrepid2::Cubature<DeviceType,double,double> > CubatureFactory::create(CellTopoPtr cellTopo, std::vector<int> cubDegree)
{
  Teuchos::RCP< Cubature > shardsTopoCub;
  int degreeOffset = 0;
  if (cubDegree.size() == cellTopo->getDimension())
  {
    vector<int> shardsDegrees;
    shardsDegrees.insert(shardsDegrees.begin(), cubDegree.begin(),cubDegree.begin() + cellTopo->getShardsTopology().getDimension());
    degreeOffset = cellTopo->getShardsTopology().getDimension();
    shardsTopoCub = Intrepid2::DefaultCubatureFactory::create<DeviceType>(cellTopo->getShardsTopology(), shardsDegrees);
  }
  else if (cubDegree.size() == cellTopo->getTensorialDegree() + 1)
  {
    int shardsDegree = cubDegree[0];
    degreeOffset = 1;
    Teuchos::RCP< Cubature > shardsTopoCub = Intrepid2::DefaultCubatureFactory::create<DeviceType>(cellTopo->getShardsTopology(), shardsDegree);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cubDegree length must either be equal to the tensorial degree of the cellTopo plus 1 or equal to the spatial dimension of the cellTopo");
  }

  if (cellTopo->getTensorialDegree() == 0)
  {
    return shardsTopoCub;
  }
  else
  {
    shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
    
    using CubatureDirectLine = Intrepid2::CubatureDirectLineGauss<DeviceType,double,double>;
    CubatureDirectLine lineCub0(cubDegree[degreeOffset]);
    
    using CubatureTensor = Intrepid2::CubatureTensor<DeviceType,double,double>;
    
    Teuchos::RCP<CubatureTensor> cubatureTensor;
    
    // we want to create the first tensor product from shardsTopoCub and lineCub
    // it's a little complicated because of the way that CubatureTensor is constructed
    // shardsTopoCub is either some sort of CubatureDirect object, or it is itself a CubatureTensor
    CubatureTensor* shardsTopoCubAsTensor = dynamic_cast<CubatureTensor*>(shardsTopoCub.get());
    if (shardsTopoCubAsTensor)
    {
      cubatureTensor = Teuchos::rcp(new CubatureTensor(*shardsTopoCubAsTensor, lineCub0));
    }
    else
    {
      using CubatureDirect = Intrepid2::CubatureDirect<DeviceType,double,double>;
      
      CubatureDirect* cubatureDirectPtr = dynamic_cast<CubatureDirect*>(shardsTopoCub.get());
      
      TEUCHOS_TEST_FOR_EXCEPTION(cubatureDirectPtr == NULL, std::invalid_argument, "shardsTopoCub is not a subclass of CubatureDirect or CubatureTensor");
      
      CubatureDirect cubatureDirect = *cubatureDirectPtr;
      
      cubatureTensor = Teuchos::rcp(new CubatureTensor(*cubatureDirectPtr, lineCub0));
    }
    
    for (int tensorialComponent=1; tensorialComponent<cellTopo->getTensorialDegree(); tensorialComponent++)
    {
      CubatureDirectLine lineCub(cubDegree[tensorialComponent + degreeOffset]);
      cubatureTensor = Teuchos::rcp( new CubatureTensor(*cubatureTensor, lineCub) );
    }
    return cubatureTensor;
  }
}
