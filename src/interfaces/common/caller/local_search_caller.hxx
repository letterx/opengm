#ifndef LOCAL_SEARCH_CALLER_HXX_
#define LOCAL_SEARCH_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/local_search.hxx>

#include "../argument/argument.hxx"


namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LocalSearchCaller : public InferenceCallerBase<IO, GM, ACC, LocalSearchCaller<IO, GM, ACC> > {

public: 
   typedef LocalSearchWrapper<GM, ACC> LocalSearchType;
   typedef InferenceCallerBase<IO, GM, ACC, LocalSearchCaller<IO, GM, ACC> > BaseClass;
   typedef typename LocalSearchType::VerboseVisitorType VerboseVisitorType;
   typedef typename LocalSearchType::EmptyVisitorType EmptyVisitorType;
   typedef typename LocalSearchType::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   LocalSearchCaller(IO& ioIn);

protected:
   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   int dummyIntParam_;
   std::string dummyStringParam_;

   void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline LocalSearchCaller<IO, GM, ACC>::LocalSearchCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of Local Search caller...", ioIn) {

   addArgument(IntArgument<>(dummyIntParam_, "", "dummyInt", "A dummy integer parameter with default 100", 100));

   std::vector<std::string> permittedDummyStringTypes = { "FOO", "BAR", "BAZ" };
   addArgument(StringArgument<>(dummyStringParam_, "", "dummyString", "a dummy parameter with finite choices", permittedDummyStringTypes.at(0), permittedDummyStringTypes));
}

template <class IO, class GM, class ACC>
void LocalSearchCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
 
   typename LocalSearchType::Parameter parameter;
   parameter.dummyInt_ = dummyIntParam_;

   //LabelInitialType
   if(dummyStringParam_ == "FOO") {
      parameter.dummyEnum_ = LocalSearchType::Parameter::FOO;
   } else if(dummyStringParam_ == "BAR") {
      parameter.dummyEnum_ = LocalSearchType::Parameter::BAR;
   } else if(dummyStringParam_ == "BAZ") {
      parameter.dummyEnum_ = LocalSearchType::Parameter::BAZ;
   } else {
      throw RuntimeError("Unknown initial label type!");
   }

   this-> template infer<LocalSearchType, TimingVisitorType, typename LocalSearchType::Parameter>(model, output, verbose, parameter);

 
}

template <class IO, class GM, class ACC>
const std::string LocalSearchCaller<IO, GM, ACC>::name_ = "LOCAL_SEARCH";

} // namespace interface

} // namespace opengm

#endif /* ALPHAEXPANSIONFUAION_CALLER_HXX_ */
