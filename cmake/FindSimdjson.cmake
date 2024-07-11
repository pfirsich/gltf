include(FetchContent)

FetchContent_Declare(
  simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_TAG tags/v0.2.1 # 0.3.0 does not work (minor version increase)
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(simdjson)
