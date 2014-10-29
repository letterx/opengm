#pragma once
#ifndef OPENGM_SOS_UB_HXX
#define OPENGM_SOS_UB_HXX

#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "submodular-ibfs.hpp"

namespace opengm {

/// Alpha-Expansion-Fusion Algorithm
/// uses the code of Alexander Fix to reduce the higer order moves to binary pairwise problems which are solved by QPBO as described in
/// Alexander Fix, Artinan Gruber, Endre Boros, Ramin Zabih:  A Graph Cut Algorithm for Higher Order Markov Random Fields, ICCV 2011
///
/// Corresponding author: Joerg Hendrik Kappes
///
/// \ingroup inference
template<class GM, class ACC>
class SoS_UBWrapper : public Inference<GM, ACC>
{
public:
   typedef GM GraphicalModelType; 
   typedef ACC AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<SoS_UBWrapper<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<SoS_UBWrapper<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<SoS_UBWrapper<GM,ACC> >  TimingVisitorType;

   struct Parameter {
      enum LabelingIntitialType {DEFAULT_LABEL, RANDOM_LABEL, LOCALOPT_LABEL, EXPLICIT_LABEL};

      Parameter
      (
         const size_t maxNumberOfSteps  = 1000
      )
      :  maxNumberOfSteps_(maxNumberOfSteps),
         labelInitialType_(DEFAULT_LABEL),
         randSeedLabel_(0),
         label_(),
         ubFn_(SoSGraph::UBfn::pairwise)
      {}

      size_t maxNumberOfSteps_;
      LabelingIntitialType labelInitialType_;
      unsigned int randSeedLabel_;
      std::vector<LabelType> label_;
      SoSGraph::UBfn ubFn_;
   };

   SoS_UBWrapper(const GraphicalModelType&, Parameter para = Parameter());

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
   void setInitialLabel(std::vector<LabelType>& l);
   void setInitialLabelLocalOptimal();
   void setInitialLabelRandom(unsigned int);
};

template<class GM, class ACC>
inline std::string
SoS_UBWrapper<GM, ACC>::name() const
{
   return "SoS_UB";
}

template<class GM, class ACC>
inline const typename SoS_UBWrapper<GM, ACC>::GraphicalModelType&
SoS_UBWrapper<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
template<class StateIterator>
inline void
SoS_UBWrapper<GM, ACC>::setState
(
   StateIterator begin,
   StateIterator end
)
{
   label_.assign(begin, end);
}

template<class GM, class ACC>
inline void
SoS_UBWrapper<GM,ACC>::setStartingPoint
(
   typename std::vector<typename SoS_UBWrapper<GM,ACC>::LabelType>::const_iterator begin
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
SoS_UBWrapper<GM, ACC>::SoS_UBWrapper
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

   if(parameter_.labelInitialType_ == Parameter::RANDOM_LABEL) {
      setInitialLabelRandom(parameter_.randSeedLabel_);
   }
   else if(parameter_.labelInitialType_ == Parameter::LOCALOPT_LABEL) {
      setInitialLabelLocalOptimal();
   }
   else if(parameter_.labelInitialType_ == Parameter::EXPLICIT_LABEL) {
      setInitialLabel(parameter_.label_);
   }
   else{
      label_.resize(gm_.numberOfVariables(), 0);
   }
}

// reset assumes that the structure of
// the graphical model has not changed
template<class GM, class ACC>
inline void
SoS_UBWrapper<GM, ACC>::reset() {
   if(parameter_.labelInitialType_ == Parameter::RANDOM_LABEL) {
      setInitialLabelRandom(parameter_.randSeedLabel_);
   }
   else if(parameter_.labelInitialType_ == Parameter::LOCALOPT_LABEL) {
      setInitialLabelLocalOptimal();
   }
   else if(parameter_.labelInitialType_ == Parameter::EXPLICIT_LABEL) {
      setInitialLabel(parameter_.label_);
   }
   else{
      std::fill(label_.begin(),label_.end(),0);
   }
}

template<class GM, class ACC>
inline InferenceTermination
SoS_UBWrapper<GM, ACC>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}

template<class GM, class ACC>
template<class Visitor>
InferenceTermination
SoS_UBWrapper<GM, ACC>::infer
(
   Visitor& visitor
)
{
   visitor.begin(*this);

   SubmodularIBFSParams params;
   params.ub = parameter_.ubFn_;
   SubmodularIBFS solver{params};
   solver.AddNode(gm_.numberOfVariables());

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
   solver.Solve();
   for (IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
      int label = solver.GetLabel(i);
      label_[i] = label;
   }
     
   OPENGM_ASSERT(gm_.numberOfVariables() == label_.size());
   visitor(*this);
   visitor.end(*this);
   return NORMAL; 
}

template<class GM, class ACC>
inline InferenceTermination
SoS_UBWrapper<GM, ACC>::arg
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

template<class GM, class ACC>
inline void
SoS_UBWrapper<GM, ACC>::setInitialLabel
(
   std::vector<LabelType>& l
) {
   label_.resize(gm_.numberOfVariables());
   if(l.size() == label_.size()) {
      for(size_t i=0; i<l.size();++i) {
         if(l[i]>=gm_.numberOfLabels(i)) return;
      }
      for(size_t i=0; i<l.size();++i) {
         label_[i] = l[i];
      }
   }
}

template<class GM, class ACC>
inline void
SoS_UBWrapper<GM, ACC>::setInitialLabelLocalOptimal() {
   label_.resize(gm_.numberOfVariables(), 0);
   std::vector<size_t> accVec;
   for(size_t i=0; i<gm_.numberOfFactors();++i) {
      if(gm_[i].numberOfVariables()==1) {
         std::vector<size_t> state(1, 0);
         ValueType value = gm_[i](state.begin());
         for(state[0]=1; state[0]<gm_.numberOfLabels(i); ++state[0]) {
            if(AccumulationType::bop(gm_[i](state.begin()), value)) {
               value = gm_[i](state.begin());
               label_[i] = state[0];
            }
         }
      }
   }
}

template<class GM, class ACC>
inline void
SoS_UBWrapper<GM, ACC>::setInitialLabelRandom
(
   unsigned int seed
) {
   srand(seed);
   label_.resize(gm_.numberOfVariables());
   for(size_t i=0; i<gm_.numberOfVariables();++i) {
      label_[i] = rand() % gm_.numberOfLabels(i);
   }
}
}

#endif // #ifndef OPENGM_ALPHAEXPANSIONFUSION_HXX
