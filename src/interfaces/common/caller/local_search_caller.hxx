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

   int rows_;
   int cols_;
   bool debug_;

   void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline LocalSearchCaller<IO, GM, ACC>::LocalSearchCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of Local Search caller...", ioIn) {

   addArgument(IntArgument<>(rows_, "", "rows", "Number of rows in image with default 100", 100));
   addArgument(IntArgument<>(cols_, "", "cols", "Number of cols in image with default 100", 100));
   addArgument(BoolArgument(debug_, "", "debug", "Boolean for whether or not to include debug effects"));
}

template <class IO, class GM, class ACC>
void LocalSearchCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
 
   typename LocalSearchType::Parameter parameter;
   parameter.rows = rows_;
   parameter.cols = cols_;
   parameter.debug = debug_;

   this-> template infer<LocalSearchType, TimingVisitorType, typename LocalSearchType::Parameter>(model, output, verbose, parameter);

 
}

template <class IO, class GM, class ACC>
const std::string LocalSearchCaller<IO, GM, ACC>::name_ = "LOCAL_SEARCH";

} // namespace interface

} // namespace opengm

#endif /* ALPHAEXPANSIONFUAION_CALLER_HXX_ */
