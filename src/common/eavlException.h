// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_EXCEPTION_H
#define EAVL_EXCEPTION_H

#include "STL.h"

// ****************************************************************************
// Class:  eavlException
//
// Purpose:
///   A base class for exception handling.
//
// Programmer:  Jeremy Meredith
// Creation:    July 16, 2012
//
// Modifications:
// ****************************************************************************
class eavlException
{
  protected:
    std::string message;
    std::string type;
    std::string file;
    int         line;
    std::string func;
  public:
    eavlException(const std::string &msg = "unknown")
        : message(msg),
          type("eavlException"),
          file("unknown"),line(-1),func("unknown")
    {
    }

    void SetTypeAndMessage(const std::string type_,
                           const std::string msg = "unknown")
    {
        type = type_;
        message = msg;
    }

    void SetThrowLocation(const std::string &file_,
                          int                line_,
                          const std::string &func_)
    {
        file = file_;
        line = line_;
        func = func_;
    }

    const std::string &GetMessage() const { return message; }

    std::string GetErrorText() const
    {
        ostringstream out;
        out << "Error: " << message << "."
            << "  " << type
            << " (occurred at " << file << ":" << line
            << " in "<< func << ")" << endl;
        return out.str();
    }

    const std::string &GetType() const { return type; }
    const std::string &GetFile() const { return file; }
    int                GetLine() const { return line; }
    const std::string &GetFunc() const { return func; }    
};

// copied from boost/current_function.hpp
#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
# define EAVL_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
# define EAVL_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
# define EAVL_CURRENT_FUNCTION __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
# define EAVL_CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
# define EAVL_CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
# define EAVL_CURRENT_FUNCTION __func__
#else
# define EAVL_CURRENT_FUNCTION "(unknown)"
#endif

#include "eavlPlatform.h"

#endif
