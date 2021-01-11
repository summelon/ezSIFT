## ezSIFT: An easy-to-use stanalone SIFT library 

[![License][license-img]][license-url]

This repository is forked from https://github.com/robertwgh/ezSIFT.  
It is tested on ubuntu 18.04 only.  
To reproduce our result, run the examples below.

### Dependencies
- Cmake: 3.9
- GCC: 8.4.0
- NVCC: 10.2

### How to build
Follow the following instructions:
```Bash
cd platforms/desktop/
mkdir build
cd build
cmake ..
make
```
Then you can find the built binary under `build/bin` directory.  
__Remember to rebuild after each checkout!!__

#### OpenMP + SIMD(AVX2)
```bash
git checkout c6bc9b7
./image_match img1.pgm img2.pgm 4 # 4 for 4 threads
```
#### OpenMP + SIMD(NEON)
```bash
git checkout 1a5e01e
./image_match img1.pgm img2.pgm 4 # 4 for 4 threads
```
#### CUDA w/ Separable Convolution
```bash
git checkout ddff0ab
./image_match img1.pgm img2.pgm
```



### License

    Copyright 2013 Guohui Wang

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


[license-url]: https://github.com/robertwgh/ezSIFT/blob/master/LICENSE
[license-img]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
