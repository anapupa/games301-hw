#if(TARGET tinyAD)
#    return()
#endif()

include(FetchContent)
FetchContent_Declare(
        tinyad
        GIT_REPOSITORY https://github.com/patr-schm/TinyAD
        GIT_TAG 60c15e889b569423f8e94ab0ddec312950d96ad1
)
FetchContent_MakeAvailable(tinyad)
