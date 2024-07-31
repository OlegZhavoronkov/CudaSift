#include <cudasift/cudaSiftException.h>
#include <cuda_runtime.h>
#include <sstream>

namespace cudasift
{
CudaSiftException::CudaSiftException( const std::string& msg )
    :   std::runtime_error( std::string( "CudaSiftException: " ) + msg ) ,
        _file( ) ,
        _line(-1)
{
    _msg = std::string("CudaSiftException: ") + msg;
}

CudaSiftException::CudaSiftException( const std::string& file , const int line , const std::string& msg )
    :   std::runtime_error( "CudaSiftException:  " + msg ) ,
        _file( file) ,
        _line(line)
{
}

CudaSiftException::CudaSiftException( const std::string& msg , const FromChild& )
    :   std::runtime_error( msg ) ,
        _msg(msg),
        _file( ) ,
        _line( -1 )
{
}

CudaSiftException::CudaSiftException( const std::string& file , const int line , const std::string& msg , const FromChild& )
    :   std::runtime_error( msg ) ,
        _msg(msg),
        _file( file ) ,
        _line( line )
{
    
}

const char* CudaSiftException::what( ) const noexcept
{
    return _msg.c_str( );
}

CudaSiftGpuException::CudaSiftGpuException( const std::string& msg )
    :   CudaSiftException( std::string( "CudaSiftGpuException: " ) + msg , CudaSiftException::FromChild{} ) ,
        _cudaErr( cudaError::cudaErrorAssert ) 
{
    //_cudaErr = cudaError::cudaErrorAssert;
}

CudaSiftGpuException::CudaSiftGpuException( int cudaErr , const char* file , const int line , const char* msg )
    : CudaSiftException(    file ,
                            line ,
                            std::string( "CudaSiftGpuException: line#" ) +
                            std::to_string( line ) +
                            std::string( " file " ) +
                            file +
                            " cuda error " +
                            std::to_string( cudaErr ) +
                            " " +
                            msg ,
                            CudaSiftException::FromChild{} ) ,
        _cudaErr( cudaErr ) ,
        _msgGpu(msg)
{
    
}

const char* CudaSiftGpuException::what( ) const noexcept
{
    cudaError_t err = static_cast< cudaError_t >( _cudaErr );
    if (err != cudaError::cudaSuccess)
    {
        std::stringstream str;
        str << "safeCall() Runtime API error in file <" << _file
            << "> , line " << _line << "\n"
            << _msgGpu << "\n"
            << cudaGetErrorString( err ) << "\n"
            << "(cudaError " << err
            << " )\n";
        return str.str( ).c_str( );
    }
    return CudaSiftException::what( );
}


}