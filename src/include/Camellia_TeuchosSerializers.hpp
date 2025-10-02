//
//  Camellia_TeuchosSerializers.hpp
//  Camellia
//
//  Created by Roberts, Nathan V. on 9/30/25.
//

#ifndef Camellia_TeuchosSerializers_hpp
#define Camellia_TeuchosSerializers_hpp

#include <Teuchos_Serializer.hpp>

#include "TypeDefs.h"

template<typename Ordinal>
class Teuchos::SerializationTraits<Ordinal,std::pair<int,int>>
  : public Teuchos::DirectSerializationTraits<Ordinal,std::pair<int,int>>
{};

template<typename Ordinal>
class Teuchos::SerializationTraits<Ordinal,std::tuple<Camellia::IndexType,int,double>>
  : public Teuchos::DirectSerializationTraits<Ordinal,std::tuple<Camellia::IndexType,int,double>>
{};

template<typename Ordinal>
class Teuchos::SerializationTraits<Ordinal,std::tuple<Camellia::IndexType,int,int,double>>
  : public Teuchos::DirectSerializationTraits<Ordinal,std::tuple<Camellia::IndexType,int,int,double>>
{};

#endif /* Camellia_TeuchosSerializers_hpp */
