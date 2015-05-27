#include "TestBilinearFormDx.h"

#include "VarFactory.h"

BFPtr TestBilinearFormDx::bf()
{
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr u = vf->fieldVar("u",HGRAD);
  VarPtr v = vf->testVar("v",HGRAD);
  BFPtr bf = BF::bf(vf);
  bf->addTerm(u->dx(), v->dx());

  return bf;
}
