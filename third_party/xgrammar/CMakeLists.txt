# Static C++ core of xgrammar v0.2.7 (see README.ninfer.md). The source list is explicit: adding a
# file is a build-boundary decision, not a recursive-glob side effect.
add_library(ninfer_third_party_xgrammar STATIC
  cpp/compiled_grammar.cc
  cpp/config.cc
  cpp/earley_parser.cc
  cpp/fsm.cc
  cpp/fsm_builder.cc
  cpp/grammar.cc
  cpp/grammar_builder.cc
  cpp/grammar_compiler.cc
  cpp/grammar_functor.cc
  cpp/grammar_matcher.cc
  cpp/grammar_parser.cc
  cpp/grammar_printer.cc
  cpp/json_schema_converter.cc
  cpp/json_schema_converter_ext.cc
  cpp/lark_converter.cc
  cpp/regex_converter.cc
  cpp/structural_tag.cc
  cpp/suffix_automata.cc
  cpp/support/logging.cc
  cpp/support/recursion_guard.cc
  cpp/testing.cc
  cpp/tokenizer_info.cc
  ninfer_hooks/logging_hooks.cc)
add_library(xgrammar::xgrammar ALIAS ninfer_third_party_xgrammar)

# Upstream builds its core as C++17; the rest of NInfer builds as C++20.
set_target_properties(ninfer_third_party_xgrammar PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ninfer_third_party_xgrammar
  PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0 XGRAMMAR_ENABLE_INTERNAL_CHECK=0
  PRIVATE XGRAMMAR_LOG_CUSTOMIZE=1)
target_include_directories(ninfer_third_party_xgrammar
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ninfer_third_party_xgrammar SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include)
target_link_libraries(ninfer_third_party_xgrammar PUBLIC Threads::Threads)
