REM Build command for Windows (MSVC + OpenMP)
REM Update the Python and pybind11 include/library paths to match your local installation.
REM Author: Xuan Tung VU

cl /O2 /EHsc /openmp /std:c++17 /LD ViabilityKernelCPU.cpp CaptureBasinCPU.cpp ROptionSetsComputer.cpp ^
  /I "C:\ProgramData\anaconda3\Include" ^
  /I "C:\Users\xtvu\AppData\Roaming\Python\Python312\site-packages\pybind11\include" ^
  /link /OUT:viability.cp312-win_amd64.pyd "C:\ProgramData\anaconda3\libs\python312.lib"
