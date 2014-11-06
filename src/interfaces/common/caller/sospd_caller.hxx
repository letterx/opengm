#ifndef SOSPD_CALLER_HXX_
#define SOSPD_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/sospd.hxx>

#include "../argument/argument.hxx"


namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class SoSPDCaller : public InferenceCallerBase<IO, GM, ACC, SoSPDCaller<IO, GM, ACC> > {

public: 
   typedef SoSPDWrapper<GM, ACC> SoSPDType;
   typedef InferenceCallerBase<IO, GM, ACC, SoSPDCaller<IO, GM, ACC> > BaseClass;
   typedef typename SoSPDType::VerboseVisitorType VerboseVisitorType;
   typedef typename SoSPDType::EmptyVisitorType EmptyVisitorType;
   typedef typename SoSPDType::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   SoSPDCaller(IO& ioIn);

protected:
    typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   size_t maxNumberOfSteps_;
   size_t randSeedOrder_;
   size_t randSeedLabel_;
   std::vector<typename GM::LabelType> labelOrder_;
   std::vector<typename GM::LabelType> label_;
   std::string desiredLabelInitialType_;
   std::string desiredUBType_;
   std::string desiredOrderType_;
   std::string desiredProposalType_;
 
   void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline SoSPDCaller<IO, GM, ACC>::SoSPDCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of Alpha-Expansion-Fusion caller...", ioIn) {
   // maxIt
   addArgument(Size_TArgument<>(maxNumberOfSteps_, "", "maxIt", "Maximum number of iterations.", (size_t)1000));
   // labelInitialType
   std::vector<std::string> permittedLabelInitialTypes;
   permittedLabelInitialTypes.push_back("DEFAULT");
   permittedLabelInitialTypes.push_back("RANDOM");
   permittedLabelInitialTypes.push_back("LOCALOPT");
   permittedLabelInitialTypes.push_back("EXPLICIT");
   addArgument(StringArgument<>(desiredLabelInitialType_, "", "labelInitialType", "select the desired initial label", permittedLabelInitialTypes.at(0), permittedLabelInitialTypes));
   // orderType
   std::vector<std::string> permittedOrderTypes;
   permittedOrderTypes.push_back("DEFAULT");
   permittedOrderTypes.push_back("RANDOM");
   permittedOrderTypes.push_back("EXPLICIT");
   addArgument(StringArgument<>(desiredOrderType_, "", "orderType", "select the desired order", permittedOrderTypes.at(0), permittedOrderTypes));
   addArgument(Size_TArgument<>(randSeedOrder_, "", "randSeedOrder", "Add description for randSeedOrder here!!!!.", (size_t)0));
   addArgument(Size_TArgument<>(randSeedLabel_, "", "randSeedLabel", "Add description for randSeedLabel here!!!!.", (size_t)0));
   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(labelOrder_, "", "labelorder", "location of the file containing a vector which specifies the desired label order", false));
   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(label_, "", "label", "location of the file containing a vector which specifies the desired label", false));
   // permittedUBTypes
   std::vector<std::string> permittedUBTypes;
   for (const auto& tuple : SoSGraph::ubParamList) {
       permittedUBTypes.push_back(std::get<1>(tuple));
   }
   addArgument(StringArgument<>(desiredUBType_, "", "ubType", "Select which upper bound to use", permittedUBTypes.at(0), permittedUBTypes));
   // proposalType
   std::vector<std::string> permittedProposalTypes = { "aexp", "blur", "grad" };
   addArgument(StringArgument<>(desiredProposalType_, "", "proposal", "Select which proposal generator to use", permittedProposalTypes.at(0), permittedProposalTypes));

}

template <class IO, class GM, class ACC>
void SoSPDCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
 
   typename SoSPDType::Parameter parameter;
   parameter.maxNumberOfSteps_ = maxNumberOfSteps_;
   parameter.randSeedOrder_ = randSeedOrder_;
   parameter.randSeedLabel_ = randSeedLabel_;
   parameter.labelOrder_ = labelOrder_;
   parameter.label_ = label_;

   //LabelInitialType
   if(desiredLabelInitialType_ == "DEFAULT") {
      parameter.labelInitialType_ = SoSPDType::Parameter::DEFAULT_LABEL;
   } else if(desiredLabelInitialType_ == "RANDOM") {
      parameter.labelInitialType_ = SoSPDType::Parameter::RANDOM_LABEL;
   } else if(desiredLabelInitialType_ == "LOCALOPT") {
      parameter.labelInitialType_ = SoSPDType::Parameter::LOCALOPT_LABEL;
   } else if(desiredLabelInitialType_ == "EXPLICIT") {
      parameter.labelInitialType_ = SoSPDType::Parameter::EXPLICIT_LABEL;
   } else {
      throw RuntimeError("Unknown initial label type!");
   }

   //orderType
   if(desiredOrderType_ == "DEFAULT") {
      parameter.orderType_ = SoSPDType::Parameter::DEFAULT_ORDER;
   } else if(desiredOrderType_ == "RANDOM") {
      parameter.orderType_ = SoSPDType::Parameter::RANDOM_ORDER;
   } else if(desiredOrderType_ == "EXPLICIT") {
      parameter.orderType_ = SoSPDType::Parameter::EXPLICIT_ORDER;
   } else {
      throw RuntimeError("Unknown order type!");
   }

   // UBType
   bool argFound = false;
   for (const auto& tuple : SoSGraph::ubParamList) {
       if (desiredUBType_ == std::get<1>(tuple)) {
           argFound = true;
           parameter.ubFn_ = std::get<0>(tuple);
       }
   }
   if (!argFound)
      throw RuntimeError("Unknown UB type!");

   // ProposalType
   if (desiredProposalType_ == "aexp") {
       parameter.proposalType_ = SoSPDType::Parameter::AEXP;
   } else if (desiredProposalType_ == "blur") {
       parameter.proposalType_ = SoSPDType::Parameter::BLUR;
   } else if (desiredProposalType_ == "grad") {
       parameter.proposalType_ = SoSPDType::Parameter::GRAD;
   } else {
      throw RuntimeError("Unknown proposal type!");
   } 

   this-> template infer<SoSPDType, TimingVisitorType, typename SoSPDType::Parameter>(model, output, verbose, parameter);

 
}

template <class IO, class GM, class ACC>
const std::string SoSPDCaller<IO, GM, ACC>::name_ = "SOSPD";

} // namespace interface

} // namespace opengm

#endif /* ALPHAEXPANSIONFUAION_CALLER_HXX_ */
