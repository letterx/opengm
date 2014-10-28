#pragma once
#ifndef OPENGM_SOSPD_HXX
#define OPENGM_SOSPD_HXX

#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "sospd.hpp"

namespace opengm {

/// Alpha-Expansion-Fusion Algorithm
/// uses the code of Alexander Fix to reduce the higer order moves to binary pairwise problems which are solved by QPBO as described in
/// Alexander Fix, Artinan Gruber, Endre Boros, Ramin Zabih:  A Graph Cut Algorithm for Higher Order Markov Random Fields, ICCV 2011
///
/// Corresponding author: Joerg Hendrik Kappes
///
/// \ingroup inference
template<class GM, class ACC>
class SoSPD : public Inference<GM, ACC>
{
public:
   typedef GM GraphicalModelType; 
   typedef ACC AccumulationType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<SoSPD<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<SoSPD<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<SoSPD<GM,ACC> >  TimingVisitorType;

   struct Parameter {
      enum LabelingIntitialType {DEFAULT_LABEL, RANDOM_LABEL, LOCALOPT_LABEL, EXPLICIT_LABEL};
      enum OrderType {DEFAULT_ORDER, RANDOM_ORDER, EXPLICIT_ORDER};

      Parameter
      (
         const size_t maxNumberOfSteps  = 1000
      )
      :  maxNumberOfSteps_(maxNumberOfSteps),
         labelInitialType_(DEFAULT_LABEL),
         orderType_(DEFAULT_ORDER),
         randSeedOrder_(0),
         randSeedLabel_(0),
         labelOrder_(),
         label_()
      {}

      size_t maxNumberOfSteps_;
      LabelingIntitialType labelInitialType_;
      OrderType orderType_;
      unsigned int randSeedOrder_;
      unsigned int randSeedLabel_;
      std::vector<LabelType> labelOrder_;
      std::vector<LabelType> label_;
   };

   SoSPD(const GraphicalModelType&, Parameter para = Parameter());

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
   std::vector<LabelType> label_;
   std::vector<LabelType> labelList_;
   size_t maxState_;
   size_t alpha_;
   size_t counter_;
   void incrementAlpha();
   void setLabelOrder(std::vector<LabelType>& l);
   void setLabelOrderRandom(unsigned int);
   void setInitialLabel(std::vector<LabelType>& l);
   void setInitialLabelLocalOptimal();
   void setInitialLabelRandom(unsigned int);
};

template<class GM, class ACC>
inline std::string
SoSPD<GM, ACC>::name() const
{
   return "SoSPD";
}

template<class GM, class ACC>
inline const typename SoSPD<GM, ACC>::GraphicalModelType&
SoSPD<GM, ACC>::graphicalModel() const
{
   return gm_;
}

template<class GM, class ACC>
template<class StateIterator>
inline void
SoSPD<GM, ACC>::setState
(
   StateIterator begin,
   StateIterator end
)
{
   label_.assign(begin, end);
}

template<class GM, class ACC>
inline void
SoSPD<GM,ACC>::setStartingPoint
(
   typename std::vector<typename SoSPD<GM,ACC>::LabelType>::const_iterator begin
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
SoSPD<GM, ACC>::SoSPD
(
   const GraphicalModelType& gm,
   Parameter para
)
:  gm_(gm),
   parameter_(para),
   maxState_(0)
{
   for(size_t j=0; j<gm_.numberOfFactors(); ++j) {
      if(gm_[j].numberOfVariables() > maxOrder_) {
         throw RuntimeError("This implementation of Alpha-Expansion-Fusion supports only factors of this order! Increase the constant maxOrder_!");
      }
   }
   for(size_t i=0; i<gm_.numberOfVariables(); ++i) {
      size_t numSt = gm_.numberOfLabels(i);
      if(numSt > maxState_) {
         maxState_ = numSt;
      }
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


   if(parameter_.orderType_ == Parameter::RANDOM_ORDER) {
      setLabelOrderRandom(parameter_.randSeedOrder_);
   }
   else if(parameter_.orderType_ == Parameter::EXPLICIT_ORDER) {
      setLabelOrder(parameter_.labelOrder_);
   }
   else{
      labelList_.resize(maxState_);
      for(size_t i=0; i<maxState_; ++i)
         labelList_[i] = i;
   }

   counter_ = 0;
   alpha_   = labelList_[counter_];
}

// reset assumes that the structure of
// the graphical model has not changed
template<class GM, class ACC>
inline void
SoSPD<GM, ACC>::reset() {
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


   if(parameter_.orderType_ == Parameter::RANDOM_ORDER) {
      setLabelOrderRandom(parameter_.randSeedOrder_);
   }
   else if(parameter_.orderType_ == Parameter::EXPLICIT_ORDER) {
      setLabelOrder(parameter_.labelOrder_);
   }
   else{
      for(size_t i=0; i<maxState_; ++i)
         labelList_[i] = i;
   }
   counter_ = 0;
   alpha_   = labelList_[counter_];
}

template<class GM, class ACC>
inline InferenceTermination
SoSPD<GM, ACC>::infer()
{
   EmptyVisitorType visitor;
   return infer(visitor);
}

template<class GM, class ACC>
template<class Visitor>
InferenceTermination
SoSPD<GM, ACC>::infer
(
   Visitor& visitor
)
{
   bool exitInf = false;
   size_t it = 0;
   size_t countUnchanged = 0;
//   size_t numberOfVariables = gm_.numberOfVariables();
//   std::vector<size_t> variable2Node(numberOfVariables);
   //ValueType energy = gm_.evaluate(label_);
   //visitor.begin(*this,energy,this->bound(),0);
   visitor.begin(*this);
   while(it++ < parameter_.maxNumberOfSteps_ && countUnchanged < maxState_ && exitInf == false) {
      // DO MOVE 
      unsigned int maxNumAssignments = 1 << maxOrder_;
      std::vector<ValueType> coeffs(maxNumAssignments);
      std::vector<LabelType> cliqueLabels(maxOrder_);

      HigherOrderEnergy<ValueType, maxOrder_> hoe;
      hoe.AddVars(gm_.numberOfVariables());
      for(IndexType f=0; f<gm_.numberOfFactors(); ++f){
         IndexType size = gm_[f].numberOfVariables();
         if (size == 0) {
            continue;
         } else if (size == 1) {
            IndexType var = gm_[f].variableIndex(0);
            ValueType e0 = gm_[f](&label_[var]);
            ValueType e1 = gm_[f](&alpha_);
            hoe.AddUnaryTerm(var, e1 - e0);
         } else {

            // unsigned int numAssignments = std::pow(2,size);
            unsigned int numAssignments = 1 << size;
            // -- // ValueType coeffs[numAssignments];
            for (unsigned int subset = 1; subset < numAssignments; ++subset) {
               coeffs[subset] = 0;
            }
            // For each boolean assignment, get the clique energy at the 
            // corresponding labeling
            // -- // LabelType cliqueLabels[size];
            for(unsigned int assignment = 0;  assignment < numAssignments; ++assignment){
               for (unsigned int i = 0; i < size; ++i) {
                  // only true for each second assigment?!?
                  //if (    assignment%2 ==  (std::pow(2,i))%2  )
                  if (assignment & (1 << i)) { 
                     cliqueLabels[i] = alpha_;
                  } else {
                     cliqueLabels[i] = label_[gm_[f].variableIndex(i)];
                  }
               }
               ValueType energy = gm_[f](cliqueLabels.begin());
               for (unsigned int subset = 1; subset < numAssignments; ++subset){
                  // if (assigment%2 != subset%2)
                  if (assignment & ~subset) {
                     continue;
                  } 
                  //(assigment%2 == subset%2)
                  else {
                     int parity = 0;
                     for (unsigned int b = 0; b < size; ++b) {
                        parity ^=  (((assignment ^ subset) & (1 << b)) != 0);
                     }
                     coeffs[subset] += parity ? -energy : energy;
                  }
               }
            }
            typename HigherOrderEnergy<ValueType, maxOrder_> ::VarId vars[maxOrder_];
            for (unsigned int subset = 1; subset < numAssignments; ++subset) {
               int degree = 0;
               for (unsigned int b = 0; b < size; ++b) {
                  if (subset & (1 << b)) {
                     vars[degree++] = gm_[f].variableIndex(b);
                  }
               }
               std::sort(vars, vars+degree);
               hoe.AddTerm(coeffs[subset], degree, vars);
            }
         }
      }  
      kolmogorov::qpbo::QPBO<ValueType>  qr(gm_.numberOfVariables(), 0); 
      hoe.ToQuadratic(qr);
      qr.Solve();
      IndexType numberOfChangedVariables = 0;
      for (IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
         int label = qr.GetLabel(i);
         if (label == 1) {
            label_[i] = alpha_;
            ++numberOfChangedVariables;
         } 
      }
      
      OPENGM_ASSERT(gm_.numberOfVariables() == label_.size());
      //ValueType energy2 = gm_.evaluate(label_);
      if(numberOfChangedVariables>0){
         //energy=energy2;
         countUnchanged = 0;
      }else{
         ++countUnchanged;
      }
      //visitor(*this,energy2,this->bound(),"alpha",alpha_);
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         exitInf = true;
      }
      // OPENGM_ASSERT(!AccumulationType::ibop(energy2, energy));
      incrementAlpha();
      OPENGM_ASSERT(alpha_ < maxState_);
   } 
   //visitor.end(*this,energy,this->bound(),0);
   visitor.end(*this);
   return NORMAL; 
}

template<class GM, class ACC>
inline InferenceTermination
SoSPD<GM, ACC>::arg
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
SoSPD<GM, ACC>::setLabelOrder
(
   std::vector<LabelType>& l
) {
   if(l.size() == maxState_) {
      labelList_=l;
   }
}

template<class GM, class ACC>
inline void
SoSPD<GM, ACC>::setLabelOrderRandom
(
   unsigned int seed
) {
   srand(seed);
   labelList_.resize(maxState_);
   for (size_t i=0; i<maxState_;++i) {
      labelList_[i]=i;
   }
   random_shuffle(labelList_.begin(), labelList_.end());
}

template<class GM, class ACC>
inline void
SoSPD<GM, ACC>::setInitialLabel
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
SoSPD<GM, ACC>::setInitialLabelLocalOptimal() {
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
SoSPD<GM, ACC>::setInitialLabelRandom
(
   unsigned int seed
) {
   srand(seed);
   label_.resize(gm_.numberOfVariables());
   for(size_t i=0; i<gm_.numberOfVariables();++i) {
      label_[i] = rand() % gm_.numberOfLabels(i);
   }
}

template<class GM, class ACC>
inline void
SoSPD<GM, ACC>::incrementAlpha() {
   counter_ = (counter_+1) % maxState_;
   alpha_ = labelList_[counter_];
}

} // namespace opengm

#endif // #ifndef OPENGM_ALPHAEXPANSIONFUSION_HXX
