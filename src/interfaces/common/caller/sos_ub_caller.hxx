#ifndef SOS_UB_CALLER_HXX_
#define SOS_UB_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/sos_ub.hxx>

#include "../argument/argument.hxx"


namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class SoS_UBCaller : public InferenceCallerBase<IO, GM, ACC, SoS_UBCaller<IO, GM, ACC> > {

public: 
   typedef SoS_UBWrapper<GM, ACC> SoS_UBType;
   typedef InferenceCallerBase<IO, GM, ACC, SoS_UBCaller<IO, GM, ACC> > BaseClass;
   typedef typename SoS_UBType::VerboseVisitorType VerboseVisitorType;
   typedef typename SoS_UBType::EmptyVisitorType EmptyVisitorType;
   typedef typename SoS_UBType::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   SoS_UBCaller(IO& ioIn);

protected:
   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   size_t maxNumberOfSteps_;
   size_t randSeedLabel_;
   std::string desiredLabelInitialType_;
   std::string desiredUBType_;
   std::vector<typename GM::LabelType> label_;
 
   void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline SoS_UBCaller<IO, GM, ACC>::SoS_UBCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of Alpha-Expansion-Fusion caller...", ioIn) {
   addArgument(Size_TArgument<>(maxNumberOfSteps_, "", "maxIt", "Maximum number of iterations.", (size_t)1000));
   std::vector<std::string> permittedLabelInitialTypes;
   permittedLabelInitialTypes.push_back("DEFAULT");
   permittedLabelInitialTypes.push_back("RANDOM");
   permittedLabelInitialTypes.push_back("LOCALOPT");
   permittedLabelInitialTypes.push_back("EXPLICIT");
   addArgument(StringArgument<>(desiredLabelInitialType_, "", "labelInitialType", "select the desired initial label", permittedLabelInitialTypes.at(0), permittedLabelInitialTypes));
   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(label_, "", "label", "location of the file containing a vector which specifies the desired label", false));
   std::vector<std::string> permittedUBTypes;
   for (const auto& tuple : SoSGraph::ubParamList) {
       permittedUBTypes.push_back(std::get<1>(tuple));
   }
   addArgument(StringArgument<>(desiredUBType_, "", "ubType", "Select which upper bound to use", permittedUBTypes.at(0), permittedUBTypes));

}

template <class IO, class GM, class ACC>
void SoS_UBCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
 
   typename SoS_UBType::Parameter parameter;
   parameter.maxNumberOfSteps_ = maxNumberOfSteps_;
   parameter.randSeedLabel_ = randSeedLabel_;
   parameter.label_ = label_;

   //LabelInitialType
   if(desiredLabelInitialType_ == "DEFAULT") {
      parameter.labelInitialType_ = SoS_UBType::Parameter::DEFAULT_LABEL;
   } else if(desiredLabelInitialType_ == "RANDOM") {
      parameter.labelInitialType_ = SoS_UBType::Parameter::RANDOM_LABEL;
   } else if(desiredLabelInitialType_ == "LOCALOPT") {
      parameter.labelInitialType_ = SoS_UBType::Parameter::LOCALOPT_LABEL;
   } else if(desiredLabelInitialType_ == "EXPLICIT") {
      parameter.labelInitialType_ = SoS_UBType::Parameter::EXPLICIT_LABEL;
   } else {
      throw RuntimeError("Unknown initial label type!");
   }

   // UBType
   bool ubFound = false;
   for (const auto& tuple : SoSGraph::ubParamList) {
       if (desiredUBType_ == std::get<1>(tuple)) {
           ubFound = true;
           parameter.ubFn_ = std::get<0>(tuple);
       }
   }
   if (!ubFound) {
      throw RuntimeError("Unknown UB type!");
   }

   this-> template infer<SoS_UBType, TimingVisitorType, typename SoS_UBType::Parameter>(model, output, verbose, parameter);

 
}

template <class IO, class GM, class ACC>
const std::string SoS_UBCaller<IO, GM, ACC>::name_ = "SOS_UB";

} // namespace interface

} // namespace opengm

#endif /* ALPHAEXPANSIONFUAION_CALLER_HXX_ */
