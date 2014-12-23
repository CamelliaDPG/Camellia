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

using namespace Teuchos;

using namespace Camellia;

using namespace Intrepid;

Teuchos::RCP<Intrepid::Cubature<double> > CubatureFactory::create(CellTopoPtr cellTopo, int cubDegree) {
  Teuchos::RCP<Cubature<double> > shardsTopoCub;
  int numShardsTopoCubatures = 0; // 1 or 0
  if (cellTopo->getShardsTopology().getDimension() != 0) {
    shardsTopoCub = _cubFactory.create(cellTopo->getShardsTopology(), cubDegree);
    numShardsTopoCubatures = 1;
  }
  
  if (cellTopo->getTensorialDegree() == 0) {
    return shardsTopoCub;
  } else {
    std::vector< Teuchos::RCP< Intrepid::Cubature<double> > > componentCubatures(cellTopo->getTensorialDegree() + numShardsTopoCubatures);
    if (numShardsTopoCubatures == 1) componentCubatures[0] = shardsTopoCub;
    shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );

    Teuchos::RCP<Cubature<double> > lineCub = _cubFactory.create( line_2, cubDegree);
    
    for (int tensorialComponent=0; tensorialComponent<cellTopo->getTensorialDegree(); tensorialComponent++) {
      componentCubatures[numShardsTopoCubatures + tensorialComponent] = lineCub;
    }
    return Teuchos::rcp(new CubatureTensor<double>(componentCubatures));
  }
}

Teuchos::RCP<Intrepid::Cubature<double> > CubatureFactory::create(CellTopoPtr cellTopo, std::vector<int> cubDegree) {
  Teuchos::RCP<Cubature<double> > shardsTopoCub;
  int degreeOffset = 0;
  if (cubDegree.size() == cellTopo->getDimension()) {
    vector<int> shardsDegrees;
    shardsDegrees.insert(shardsDegrees.begin(), cubDegree.begin(),cubDegree.begin() + cellTopo->getShardsTopology().getDimension());
    degreeOffset = cellTopo->getShardsTopology().getDimension();
    shardsTopoCub = _cubFactory.create(cellTopo->getShardsTopology(), shardsDegrees);
  } else if (cubDegree.size() == cellTopo->getTensorialDegree() + 1) {
    int shardsDegree = cubDegree[0];
    degreeOffset = 1;
    Teuchos::RCP<Cubature<double> > shardsTopoCub = _cubFactory.create(cellTopo->getShardsTopology(), shardsDegree);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cubDegree length must either be equal to the tensorial degree of the cellTopo plus 1 or equal to the spatial dimension of the cellTopo");
  }
  
  if (cellTopo->getTensorialDegree() == 0) {
    return shardsTopoCub;
  } else {
    std::vector< Teuchos::RCP< Intrepid::Cubature<double> > > componentCubatures(cellTopo->getTensorialDegree() + 1);
    componentCubatures[0] = shardsTopoCub;
    shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
    
    for (int tensorialComponent=0; tensorialComponent<cellTopo->getTensorialDegree(); tensorialComponent++) {
      Teuchos::RCP<Cubature<double> > lineCub = _cubFactory.create( line_2, cubDegree[tensorialComponent + degreeOffset]);
      componentCubatures[1+tensorialComponent] = lineCub;
    }
    return Teuchos::rcp(new CubatureTensor<double>(componentCubatures));
  }
}