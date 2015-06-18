#pragma once
#ifndef MULTILABEL_LOCAL_SEARCH_UB_HXX
#define MULTILABEL_LOCAL_SEARCH_UB_HXX

#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "multilabel-local-search-higher-order-energy.hpp"
#include "multilabel-local-search.hpp"

namespace opengm {

/// \ingroup inference
template<class GM, class ACC>
class MultiLabelLocalSearchWrapper : public Inference<GM, ACC>
{
public:
   typedef GM GraphicalModelType; 
   typedef ACC AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<MultiLabelLocalSearchWrapper<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<MultiLabelLocalSearchWrapper<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<MultiLabelLocalSearchWrapper<GM,ACC> >  TimingVisitorType;

   struct Parameter {
      int rows;
      int cols;
      bool debug;

      Parameter ( )
      : rows(0),
        cols(0),
        debug(false)
      {}
   };

   MultiLabelLocalSearchWrapper(const GraphicalModelType&, Parameter para = Parameter());

   std::string name() const;
   const GraphicalModelType& graphicalModel() const;
   template<class StateIterator>
      void setState(StateIterator, StateIterator);
   InferenceTermination infer();
   void reset();
   template<class Visitor>
      InferenceTermination infer(Visitor& visitor);
   void setStartingPoint(typename std::vector<LabelType>::const_iterator);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

private:
   const GraphicalModelType& gm_;
   Parameter parameter_; 
   static const size_t maxOrder_ =10;
   static constexpr double DoubleToREALScale = 10000;
   std::vector<LabelType> label_;
};

template<class GM, class ACC>
inline std::string
MultiLabelLocalSearchWrapper<GM, ACC>::name() const
{
   return "MultiLabelLocalSearch";
}

template<class GM, class ACC>
inline const typename MultiLabelLocalSearchWrapper<GM, ACC>::GraphicalModelType&
MultiLabelLocalSearchWrapper<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
template<class StateIterator>
inline void
MultiLabelLocalSearchWrapper<GM, ACC>::setState
(
   StateIterator begin,
   StateIterator end
)
{
   label_.assign(begin, end);
}

template<class GM, class ACC>
inline void
MultiLabelLocalSearchWrapper<GM,ACC>::setStartingPoint
(
   typename std::vector<typename MultiLabelLocalSearchWrapper<GM,ACC>::LabelType>::const_iterator begin
) {
   try{
      label_.assign(begin, begin+gm_.numberOfVariables());
   }
   catch(...) {
      throw RuntimeError("unsuitable starting point");
   }
}

template<class GM, class ACC>
inline
MultiLabelLocalSearchWrapper<GM, ACC>::MultiLabelLocalSearchWrapper
(
   const GraphicalModelType& gm,
   Parameter para
)
:  gm_(gm),
   parameter_(para)
{
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      if(gm_[j].numberOfVariables() > maxOrder_) {
         throw RuntimeError("This implementation of Alpha-Expansion-Fusion supports only factors of this order! Increase the constant maxOrder_!");
      }
   }
   size_t maxState = 0;
   for(size_t i=0; i<gm_.numberOfVariables(); ++i) {
      size_t numSt = gm_.numberOfLabels(i);
      if(numSt > maxState) {
         maxState = numSt;
      }
   }
   if (maxState > 2) {
       throw RuntimeError("SOS_UB can only be used on binary energies");
   }

  label_.resize(gm_.numberOfVariables(), 0);
}

// reset assumes that the structure of
// the graphical model has not changed
template<class GM, class ACC>
inline void
MultiLabelLocalSearchWrapper<GM, ACC>::reset() {
  std::fill(label_.begin(),label_.end(),0);
}

template<class GM, class ACC>
inline InferenceTermination
MultiLabelLocalSearchWrapper<GM, ACC>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}

template<class GM, class ACC>
template<class Visitor>
InferenceTermination
MultiLabelLocalSearchWrapper<GM, ACC>::infer
(
   Visitor& visitor
)
{
   visitor.begin(*this);

   typedef MultiLabelLocalSearchHigherOrderEnergy<REAL, maxOrder_> LSHOE;
   typename LSHOE::Parameter param;
   param.rows = parameter_.rows;
   param.cols = parameter_.cols;
   param.debug = parameter_.debug;
   MLLSHOE solver{param};
   solver.AddNode(gm_.numberOfVariables());
   std::cout << "start addcliques" << std::endl;
   for(IndexType f=0; f<gm_.numberOfFactors(); ++f){
       auto& c = gm_[f];
       const size_t k = c.numberOfVariables();
       const LabelType dummyLabels[] = {0, 1};
       ASSERT(k < 32);
       if (k == 0)
          continue;
       else if (k == 1) {
          IndexType var = gm_[f].variableIndex(0);
          ValueType e0 = gm_[f](&dummyLabels[0]);
          ValueType e1 = gm_[f](&dummyLabels[1]);
          solver.AddUnaryTerm(var, e0*DoubleToREALScale, e1*DoubleToREALScale);
       } else {
           std::vector<SubmodularIBFS::NodeId> nodes(c.variableIndicesBegin(), c.variableIndicesEnd());
           const uint32_t maxAssgn = 1 << k;
           std::vector<REAL> energyTable(maxAssgn, 0);
           std::vector<LabelType> cliqueLabels(k, 0);
           for (uint32_t a = 0; a < maxAssgn; ++a) {
               for (int i = 0; i < k; ++i) {
                   if (a & (1 << i)) cliqueLabels[i] = 1;
                   else cliqueLabels[i] = 0;
               }
               energyTable[a] = static_cast<REAL>(c(cliqueLabels.begin())*DoubleToREALScale);
           }
           solver.AddClique(nodes, energyTable);
       }
   }
   std::cout << "end addcliques" << std::endl;
   MultiLabelLocalSearch crf;
   solver.ToQuadratic(crf);
   crf.Solve();
   for (IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
      int label = crf.GetLabel(i);
      label_[i] = label;
   }
     
   OPENGM_ASSERT(gm_.numberOfVariables() == label_.size());
   visitor(*this);
   visitor.end(*this);
   return NORMAL; 
}

template<class GM, class ACC>
inline InferenceTermination
MultiLabelLocalSearchWrapper<GM, ACC>::arg
(
   std::vector<LabelType>& arg,
   const size_t n
) const
{
   if(n > 1) {
      return UNKNOWN;
   }
   else {
      OPENGM_ASSERT(label_.size() == gm_.numberOfVariables());
      arg.resize(label_.size());
      for(size_t i=0; i<label_.size(); ++i) {
         arg[i] = label_[i];
      }
      return NORMAL;
   }
}

}

#endif // #ifndef OPENGM_ALPHAEXPANSIONFUSION_HXX
