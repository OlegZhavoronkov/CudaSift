#pragma once
#include <string>
#include <stdexcept>



namespace cudasift
{

class CudaSiftException :public std::runtime_error
{
public:
    explicit CudaSiftException( const std::string& msg );
    CudaSiftException( const std::string& file , const int line , const std::string& msg );
    virtual const char* what( ) const noexcept override;
    virtual ~CudaSiftException( ) noexcept = default;
protected:
    std::string _msg;
    struct FromChild
    {
    };
    CudaSiftException( const std::string& msg , const FromChild& );
    CudaSiftException( const std::string& file , const int line , const std::string& msg ,const FromChild& );
    const std::string _file;
    const int _line;
};

class CudaSiftGpuException :public CudaSiftException
{
public:
    explicit CudaSiftGpuException( const std::string& msg );
    CudaSiftGpuException( int cudaErr , const char* file , const int line , const char* msg );
    //explicit CudaSiftGpuException( const char* msg );
    virtual const char* what( ) const noexcept override;
    virtual ~CudaSiftGpuException( ) noexcept = default;
protected:
    const int _cudaErr;

    const std::string _msgGpu;
};

}