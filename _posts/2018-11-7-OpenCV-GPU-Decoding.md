# Opencv 实现GPU解码

最近一直关注使用GPU解码视频方面的工作。OpenCV已经实现了GPU解码的功能，本文将介绍一下如何使用OpenCV进行GPU解码，可能遇到的BUGs和解码速度分析。

实验环境：Ubuntu 16.04,  Nvidia K40, CUDA 9.0, OpenCV 3.4, FFmepg 3.3.1,  Intel E5-2630 v3 2.4GHz

首先需要安装CUDA，安装过程请参考NVIDIA官网教程。第二步，安装FFmpeg，参考[安装教程](https://developer.nvidia.com/ffmpeg) ，注意enable-cuda，enable-cuvid等。第三步，安装OpenCV，需要注意的是，在cmake过程中，也要-D WITH_CUDA=ON, -D WITH_NVCUVID=ON. 完成这三步以后，实验环境配置完成。

代码主要参考opencv\sources\samples\gpu\video_reader.cpp略做修改，编译链接以后，运行代码会出现错误：

Segmentation fault with gpu video decoding

代码Debug以后，将定位到这一行：

```
cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
```

这个错误的主要原因在于https://github.com/opencv/opencv/issues/10201#issuecomment-379478620：

```
The segfault happens because with the dynlink_nvcuvid.h header files you have to actually do the dynamic linking (i.e. use dlopen and set the function pointers that are in dynlink_nvcuvid.h). All of the function pointer are initialized to 0 in dynlink_nvcuvid.h, so attempting to call any of those functions without loading the symbols first will result in a segfault. The Video Codec SDK samples do this in the cuvidInit() function in dynlink_nvcuvid.cpp. You can also see this done in FFmpeg.

With the latest release of the Video Codec SDK, 8.1, they have gone back to using nvcuvid.h and normal linking. CUDA 9.1 still ships with the dynlink_nvcuvid.h header, but you can get the new one by downloading it directly. The header file itself is BSD licensed so the easiest thing may be to just include the one you are going to use in the OpenCV repo.
```

所以代码需要修改如下：

首先将/usr/local/cuda/samples/3_Imaging/cudaDecodeGL/dynlink_nvcuvid.cpp 复制到当前的源码目录下。然后修改代码如下：

```
#include <iostream>                                                                             

#include "opencv2/opencv_modules.hpp"

#include <string>
#include <sys/time.h>

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include <dynlink_nvcuvid.h>
#include <dynlink_cuviddec.h>

using namespace std;

int main(int argc, const char* argv[])
{
    if (argc != 2)
        return -1;
    struct timeval start, end;
    long second, usecond;
    const std::string fname(argv[1]);
    // 添加以下代码 
    void* hHandleDriver = 0;
    CUresult cuda_res = cuInit(0, __CUDA_API_VERSION, hHandleDriver);
    if(cuda_res != CUDA_SUCCESS){
        throw exception();
    }
    cuda_res =  cuvidInit(0);
    if(cuda_res != CUDA_SUCCESS){
        throw exception();
    }
    std::cout<<"CUDA init: SUCCESS"<<endl;

    cv::cuda::printCudaDeviceInfo(0);
    // end 添加
    cv::cuda::setDevice(0);
    cv::cuda::GpuMat d_frame;
    gettimeofday(&start, NULL);
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);

    int gpu_frame_count=0;

    for (;;)
    {
        if (!d_reader->nextFrame(d_frame))
            break;
        gpu_frame_count++;
    }
    gettimeofday(&end, NULL);

    second = end.tv_sec - start.tv_sec;
    usecond = end.tv_usec - start.tv_usec;
    double time_clapsed = second + usecond / 1000000;
    cout<<"second:\t"<<second<<"usecond:\t"<<usecond<<endl;
    cout<<"time clasped\t"<<time_clapsed<<endl;
    return 0;
}

```

makefile 如下所示：

```
extract_frame:test.o
    g++ -g -o extract_frame test.o  nvcuvid.o -Wl,-rpath=/home/wyj/software/opencv-3.4.1/build/lib -L /home/wyj/software/opencv-3.4.1/build/lib -lopencv_core -lopencv_highgui -lopencv_cudev -lopencv_cudacodec -lopencv_videoio -L /usr/lib/x86_64-linux-gnu/ -lnvcuvid -lcuda -ldl

test.o:video_reader.cpp
    g++ -g -c video_reader.cpp -o test.o -I /home/wyj/software/opencv-3.4.1/build/include -I /usr/local/cuda/include

nvcuvid.o:dynlink_nvcuvid.cpp
    g++ -g -c dynlink_nvcuvid.cpp -o nvcuvid.o -I /usr/local/cuda/include

clean:
    rm -rf test.o extract_frame nvcuvid.o 
```

make 以后即可成功运行。那么这样做效果如何，真的能实现GPU加速吗？这里，作者使用一个1920*1080，长度1分20秒的mp4视频进行测试，对比以下只使用CPU的代码：

```
#include <iostream>                                                                             
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv; 


int main(int argc, char** argv)
{
    bool stop(false);
    Mat frame;
    struct timeval start, end;
    long mtime, second, usecond;
    gettimeofday(&start, NULL);
    VideoCapture capture(argv[1]);
    if(!capture.isOpened())
    {   
        return -1; 
    }   
    double rate = capture.get(CV_CAP_PROP_FPS);
    while(!stop)
    {   
        if(!capture.read(frame))
        {
            stop = true;
        }
    }   
    gettimeofday(&end, NULL);
    second = end.tv_sec - start.tv_sec;
    usecond = end.tv_usec - start.tv_usec;
    double time_clapsed = second + usecond / 1000000;
    cout<<"second:\t"<<second<<"usecond:\t"<<usecond<<endl;
    cout<<"time clasped\t"<<time_clapsed<<endl;
    return 0;
}

```

其中CPU代码top命令查看CPU使用率一直310%左右，大概OpenCV后天自动开启了多核。解码时间却令人震惊，对比结果如下：

| CPU-OpenCV | GPU-OpenCV |
| ---------- | ---------- |
| 247s       | 447s       |

使用OpenCV的GPU并没有达到想象中的加速效果，反而速度变慢。至于背后原因，我会进一步探讨。

参考链接：

https://github.com/opencv/opencv/issues/11220

https://github.com/opencv/opencv/issues/10201

https://blog.csdn.net/cdknight_happy/article/details/81660198

http://answers.opencv.org/question/178227/cvcudacodeccreatevideoreader-segfault/

http://answers.opencv.org/question/178227/cvcudacodeccreatevideoreader-segfault/