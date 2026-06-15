# correctness/perf は実GPU必要。lit ベースの IR round-trip テストは CPU 可。
add_test(NAME correctness_skip_or_run
         COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/correctness/run.py)
