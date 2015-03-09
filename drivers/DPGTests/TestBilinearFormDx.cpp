#include "TestBilinearFormDx.h"

#include "VarFactory.h"

BFPtr TestBilinearFormDx::bf() {
  VarFactory vf;
  VarPtr u = vf.fieldVar("u",HGRAD);
  VarPtr v = vf.testVar("v",HGRAD);
  BFPtr bf = BF::bf(vf);
  bf->addTerm(u->dx(), v->dx());
  
  return bf;
}