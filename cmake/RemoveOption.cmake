cmake_minimum_required(VERSION 3.20)

function (remove_specific_option_from_command_line CLINE OPTION CLINE_OUT)
    
    separate_arguments(cargs_separated NATIVE_COMMAND "${CLINE}")
    set(CLINE_OUT_INT "")
    foreach(arg IN ITEMS ${cargs_separated} )
        if(${arg} MATCHES ${OPTION})
        else()
            set(CLINE_OUT_INT "${CLINE_OUT_INT} ${arg}")
        endif()
    endforeach()
    set( ${CLINE_OUT} ${CLINE_OUT_INT} PARENT_SCOPE)
endfunction()