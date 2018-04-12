# Explainer:  API Sets for Machine Learning on the Web 

With the recent breakthroughs in deep learning and related technologies, Machine Learning (ML) 
algorithms have drastically improved in terms of accuracy, application, performance etc. While 
typically thought of as a technology only applicable to server technologies, the 
inferencing process of machine learning models can run on device as well. Development of a 
machine learning application<sup>[1](#myfootnote1)</sup> usually involves two stages: 

* The developer first *train* the model<sup>[2](#myfootnote2)</sup> by first creating a 
    skeleton framework and then 
    iterating the model with large dataset 
* The developer then port the model to production environment so that 
    it can *infer* insight from user input 

Though training typically takes place in the cloud because it requires a significant 
amount of data and 
computing power, inference can take place in the cloud or on the device. Running inference on the 
device has a number of appealing properties, such as performance boost 
due to [edge computing](https://en.wikipedia.org/wiki/Edge_computing), resistance toward poor or 
no network, and security/privacy protection, etc. 

Although platforms for native applications have all shipped APIs to support machine learning 
inference on device, similar functionality has been missing on the web platform. The explainer 
recommends an approach to address this missing functionality by providing an improved WebAssembly
package, a WebML (Web Machine Learning) API with a pre-defined set of mathematical functions 
that the platform can optimize for, and a WebNN (Web Neural Network) API that provides a high-level abstraction to run neural networks efficiently. 
Other approaches are also welcomed. After all, 
explainer is meant for sparking conversations rather than pinpointing a clear solution. 

Please feel free to submit issues or PRs to correct me or add additional points. I am excited
to hear more conversations around this topic.

## Introduction

Supporting ML inferences not only can supercharge existing applications but also can unlock new 
scenarios (see [use cases](#use-cases)). For instance, with the help of 
[Service Worker](https://github.com/w3c/ServiceWorker), applications can translate between 
languages with poor or no network. By inferring the user’s emotions 
based on user’s input (be it text, image, or video), developers can build a rich 
interactive experience. Applications on new frontiers such as Mixed Reality can become much 
"smarter."

Developers have also shown strong interests in deploying inferences in 
web applications as evidenced by the growing number of machine learning libraries that can run in 
browsers.<sup>[3](#myfootnote3)</sup> See [here](https://github.com/AngeloKai/js-ml-libraries) 
for a short list of the libraries or frameworks. [Synaptic.js](http://caza.la/synaptic/#/) and 
[webdnn](https://mil-tokyo.github.io/webdnn/) are examples of the impressive 
works developers have done to enable this scenario. 

However, though the libraries and frameworks have helped lower the barrier of development, 
developers continue facing a 
[painful development process along with bottlenecks](#developer-pain) 
because of limitations with the current platform. Developers would run into issues such as 
porting model from C to JS, large model size, performance<sup>[4](#myfootnote4)</sup>, 
memory overflow etc. 

Four existing standard efforts helped address the pain, but none of them 
is a complete solution: 

1. [__High Level JS APIs Built On Machine Learning Technology.__](#APIs-Built-On-Machine-Learning-Technologies)

    A number of high level JavaScript APIs are built on machine learning technologies, such as  
    the [Web Speech API](https://dvcs.w3.org/hg/speech-api/raw-file/tip/webspeechapi.html), 
    [Web Authentication API](https://w3c.github.io/webauthn/), 
    [Shape Detection API](https://github.com/WICG/shape-detection-api). 
    
    However, these APIs only address a very specific scenarios instead of offering 
    [extensibility](https://extensiblewebmanifesto.org/) to developers.
    For example, developers may want to detect more than the 
    [three basic types of shapes](https://github.com/WICG/shape-detection-api#overview) offered 
    by the Shape Detection API. It is also incredibly challenging to define 
    interoperability standards for these APIs because the browsers are building them 
    with different machine learning models that have different accuracy rate. And it
    will only be more challenging with the growing complexity of the models. 

1. [__WebGL.__](#WebGL) 

    The [WebGL](https://www.khronos.org/webgl/) API is the most commonly API used in the top 
    libraries because developers can create wrapper functions around shader object to run 
    matrix computations in GPUs. However, the wrappers aren't 
    developer-friendly and usually very taxing on memory. Functions that directly do matrix 
    computation would make developers' life much easier and much simpler for the platform to 
    optimize.
    
    It is true that the next version of WebGL (WebGL 3.0) could include supporting
    machine learning inferences into its charter. However, it may be difficult because inferencing 
    and drawing graphics impose different requirements to matrix computations. For example, 
    the former only needs low level of precision whereas latter most often requires high level of
    precision. Some machine learning computations are also not entirely around matrix 
    manipulations, such as [fast fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) or 
    [huffman coding](#Huffman-Coding). 

1. [__WebGPU.__](#WebGPU)

    The [WebGPU](https://webkit.org/wp-content/uploads/webgpu-api-proposal.html#api) API is 
    in development to expose modern GPU features, though the initial API proposal based on 
    [Metal](https://developer.apple.com/documentation/metalperformanceshaders) 
    appeared to be geared toward graphics rendering but rather than matrix computations. 
    
    Contrary to popular beliefs, chips other than GPUs can be much more effective at accelerating
    ML models. More and more companies have been producing 
    [ASIC (Application-Specific Integrated Circuit)](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit)
    chips for inferencing. Modern artificial neural networks also often employ convolution
    that [DSPs (Digital Signal Processors)](https://en.wikipedia.org/wiki/Digital_signal_processor)
    are designed to accelerate. 

1. [__WebAssembly.__](#WebAssembly)

    WebAssembly addresses the developer pain of porting binary package of trained ML
    models to the web platform. 
    
    However, the performance issue still lingers around because it doesn't yet support GPU and
    multi-threaded modules. Though developers can use hacks to get those supports, it does 
    mean much more extra efforts.

    WebAssembly also doesn't address the fundamental problem that the platform lacks good support 
    for direct mathematical computations that inferencing needs. In the long term, this lack of 
    support could mean an over-bloated browser runtime and multiple libraries for browsers to 
    optimize. Learning from existing practices and standard efforts such as 
    [BLAS](http://www.netlib.org/blas/), we could develop a new API of mathematical operations for
    inferencing with appropriate hardware accelerations. 

Looking at the four existing efforts and how web platforms supported similar issues such as the
existing WebAssembly + WebGL approach, the best approach forward appeared to be a combination of :
1. WebAssembly
1. A New API of optimized mathematical functions (tentatively called
WebML). 

Some may argue that this approach would still leave us with the issue of large model 
size because of large frameworks and we could reduce it down further by developing an API with a 
higher level of abstraction that models typical neural networks. After all, 
neural networks have been the 
forefront of machine learning innovations for the last few years. This "neural network" API could 
have existing structure for developers to load weights in and an easy way for nodes of the network
to communicate, such as 
[SharedBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer). 
There are already attempts at standardizing AI models, such as [onxx](https://github.com/onnx/onnx). 

This approach would roughly match [how native platforms supported them](#native-platform-support): 
* support traditional ML algorithms such as decision tree learning or Bayesian algorithms 
    by providing optimized linear algebra libraries and other optimized mathematical functions 
* support the Deep Neural Networks approach (center of attention in the recent AI boom) by 
    shipping dedicated DNNs API. If developers find the DNNs API not complex enough to 
    handle their special case, they can also use the optimize linear algebra libraries. 


 
## Use Cases
Despite the long history of machine learning research and applications, I think it is 
safe to say we are still uncovering the countless applications ML. Below illustrate 
some example use cases developers may use machine learning in front-end applications. The 
sample use cases are based on inspirations from existing demos and production sites/apps. 

### Text Translation 
A web application may wish to translate text from one language to another offline to be network 
resilient. For example, the Google Translate service trained a neural network to 
translate between languages and ported the network to its mobile app. The mobile app 
can be used offline, though translation may be better online. 

### Offline Recommendation Engine
A web application built with Service Work to be network resistant may wish to build its 
recommendation engine offline. For example, a site serving images/GIFs/video as content 
may wish to serve users smart content feed with content cached with Service Worker. 
Or a productivity application with many different features like Office may want to 
provide [Help](https://support.office.com/en-us/article/Where-is-the-product-Help-in-Office-2016-199950c6-1260-44fe-ba2e-e95968d05397) when the user is looking to know which feature they should use. 

### Object Detection from Images/Videos
A web application may wish to detect objects from images or videos. For example, 
[Baidu](https://en.wikipedia.org/wiki/Baidu) built convolutional neural networks (CNNs) 
into its mobile app so that the app can detect the primary object in the live camera feed 
and search related merchandise based on the 
result ([Mobile Deep Learning framework by Baidu](https://github.com/baidu/mobile-deep-learning)).

In addition to generic object detection, a developer may also want to tweak his/her specific 
object detection to hone in on certain areas. For instance, if the developer anticipates 
the users would show them certain types of images, they could train the models with 
only those images to get better accuracy:

* An application may train an objection detection model with only credit card number images 
to auto-detect those numbers from live camera feed. Doing so reduces friction with 
entering credit card number and increases conversion rate from basic users to premium users. 
* An application for streaming/uploading videos may wish to perform live check of the 
camera feed to ensure the user isn’t showing obscene content for law compliance purpose.
* An application can train a model with images of skin cancer and regular skin to help 
doctors detect skin cancer 
([Dermatologist-level classification of skin cancer with deep neural networks](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html?foxtrotcallback=true)).

An application may also wish to let the front-end code only to identify the primary objects 
in an image and leave the task of classification to the back-end. Object detection roughly 
includes two sub-tasks: localization and classification. The former refers to the task of 
finding the pixel mask of the primary object in an image. The latter refers to the task of 
classifying which category an image belongs. The latter typically includes a much larger 
program while the former should usually be small enough to fit in client-side code. In 
the above skin cancer recognition example, the application may want to let the front-end 
code to identify irregularities (e.g. a mole) and leave the task of classifying 
what kind of skin cancer the irregularities signal to the back-end.

### Risk Analysis 
A web application may wish to deploy a small-scale risk analysis model to determine 
whether the transaction should be pre-approved and leave the final decision to the 
full-scale risk models on the backend. Quick pre-approval improves user 
experience while reducing cost for running the model.

### Rich Interactive Experience
A web application may wish to build "smarter" user interaction models. For example, 
the application can guess the user’s emotion based on user input and proactively 
make recommendations. For example, a social media site may wish to detect 
user’s emotion when user’s typing the post and recommend the right emoji to 
use. If the user wishes to post picture alongside the post, the application 
can also recommend appropriately based on the post.

### Mixed Reality Experience 
A web application built for mixed reality platforms may wish to leverage machine learning 
to anticipate user intention and provide assistance as needed. For example, when a 
worker is learning how to operate a new machine, the application could show how 
to operate each component as the worker gazes at each component. To do so, the app 
developer will need an objection detection model that is tuned to detect the 
components because they aren't included in typical image detection 
databases like [ImageNet](http://www.image-net.org/).  

## Developer Pain 

Developers face significant pain when trying a machine learning models into their front-end 
codebase today. The below walks through a typical development 
process<sup>[5](#myfootnote5)</sup> and illustrates specific pain points: 

1. Assuming the developer already has a trained model, the developers first face the pain 
    of large model size. Most models trained with frameworks 
    such as [Tensorflow](https://www.tensorflow.org) can have file 
    size that ranges from a few hundred Mb to 
    more than one GB. Some frameworks optimizing for mobile experience shrink the model size
    down to around 20 Mb by applying techniques such as [quantization](#quantization).
    Unfortunately, the process isn't as simple as described. The techniques do run the risk
    of reducing accuracy, so developers have to keep double checking throughout the optimization
    process. 

1. After the developer combat the model size challenge, they have to face the next challenge of 
    language choice. The most well-known machine learning frameworks today don't have the option 
    of training models in JavaScript. The typical option is writing the model in python, 
    which the frameworks can compile down to executable and 
    sometimes C.<sup>[6](#myfootnote6)</sup> Developers can either transpile the model to JS or 
    re-compile it to work with WebAssembly. The former option means increased file size 
    but easier way to improve performance. Developers can easily re-write the model to be 
    multi-threaded, a feature not yet available in WebAssembly.<sup>[7](#myfootnote7)</sup> 
    Developers can also more easily accelerate the models with GPU level acceleration 
    with existing libraries of wrapper functions of WebGL. The latter would mean consistent
    file size but much harder route to improving performance, though there is a silver lining. 
    For computations that can hardly be accelerated by GPU, such as 
    [fast fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform),
    WebAssembly can accelerate them faster because they would be computed in the CPU directly,
    such as [WebDSP](https://github.com/shamadee/web-dsp).

1. After a decent model is loaded in a web page, developers could also run into issue with 
    performance and/or memory overflow issue. Without effective hardware acceleration,
    matrix operation could take a long time. Inefficient matrix representation, especially of
    [sparse matrix](#sparse-matrix) could also lead to memory overflow. If data is stored in 
    storage, frequent read and write operations could also lead to performance slow down. 

1. During the above process, developers also need to think about what device/platform they 
    plan to run on to ensure they have the right fallback options. 

Other than application developers' pain points, platform developers also would likely run into
power issues if enough machine learning applications are running on the platform. Studies and
experiences have shown that ML can easily drain power. 

Fortunately, our colleagues working on other platforms such as IoT, native apps, 
or cloud computing,
have been working on addressing similar issues so we will have some great advisors along the 
journey. 



## Existing Standards 

The below standards helped ease the developer pain but doesn't offer a complete and long-term 
solution: 

### APIs Built On Machine Learning Technologies
In the past few years, we have been adding new APIs that relies on machine 
learning technologies:

* __[Web Speech API](https://dvcs.w3.org/hg/speech-api/raw-file/tip/webspeechapi.html)__ 
    enables developers to easily convert text content to speech and speech content to text. 
    The API itself defines the standard API interface while the implementations all rely 
    on machine learning technology behind the scene. Although there are interoperability
    differences between the implementations, speech recognition technology has matured 
    enough to mask the differences. 

* __[Web Authentication API](https://w3c.github.io/webauthn/)__ lets developers
    authenticate users with strong authenticators, such as fingerprint scanners, facial 
    recognition systems, USB tokens etc. Biometric authenticators all employ 
    machine learning technologies one way or another. Although biometric 
    recognition technology has matured enough that interoperability differences are small, 
    different authenticators do have different success rate, and these differences
    have constantly been a point of contention. 

* __[Shape Detection API](https://github.com/WICG/shape-detection-api)__, a recent 
    addition to the Web Incubator CG (WICG), can be used to detect
    faces, barcodes, and text from live camera feed or still images with accelerated
    hardware. This API is a good example of a feature that uses hardware 
    acceleration ([Image Signal Processors (ISPs)](https://en.wikipedia.org/wiki/Image_processor)) 
    to run machine learning operations.  

Among other reasons, one argument for building the APIs is the underlying machine models 
are computationally expensive to run and need hardware acceleration. However, it is 
unscalable to continue adding APIs to 
the platform for the reason of computational cost. There should be a generic solution that 
can bring down the computational cost of doing machine learning on the web platform.

### WebGL
The WebGL API was designed to render 3D and 2D graphics and make use of GPUs 
behind the scene when necessary. Given that most graphic processing relies on matrix 
computation, web developers have developed 
[libraries that wrap around WebGL to accelerate matrix computation](https://github.com/AngeloKai/js-ml-libraries). 

However, such libraries are not developer-friendly and often very taxing on memory. 
Take the example of this 
[matrix multiplication method](https://github.com/waylonflinn/weblas/blob/master/index.js#L59). 
To compute matrix multiplication, the function has to:

1. instantiate two RGBA texel arrays
1. transpose one of the arrays
1. create two input textures and one output texture
1. bind input texture
1. activate a shader
1. set shader parameters
1. bind output texture
1. call ````drawElementso()```` to do the actual __multiplication__
1. unbind all the textures. 

As seen above, the code is more than complicated and causes unnecessary amount of instantiation.

The next generation of WebGL API could include more support for direct mathematical 
computation. However, one can argue that this goal is not aligned with the charter of an API 
that was designed for drawing graphics. Inferencing and drawing graphics also impose 
different requirements to matrix manipulations. For example, research in neural network 
optimization (as seen in the [quantization](#quantization) section) has demonstrated 
inferencing can work well with low precision numerical representation whereas graphics usually 
high require high precision. The working group may have a hard time reconciling the differences. 

### WebGPU
[WebGPU](https://webkit.org/wp-content/uploads/webgpu-api-proposal.html#api) API is a new 
incubating API that aims at exposing modern GPU features. Its initial API set is a 
derivation of the Metal language. A prototype for the API has landed in WebKit.

Although the API aims at exposing low-level GPU functionalities, its initial API set is primarily 
geared toward graphics rendering and not direct mathematical computation that is more wanted by
the machine learning community. Additionally, research has also shown 
that while GPU accelerates computing, other chips can be used to accelerate inferencing. 
Companies have produced 
[ASIC](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit) chips 
for either on-device or in-cloud inferencing, such as such as 
[Movidius' (an Intel company) Myriad VPU](https://www.movidius.com/technology), 
the [IBM’s TrueNorth chips](https://www.ibm.com/blogs/research/2016/09/deep-learning-possible-embedded-systems-thanks-truenorth/), 
or [Intel’s Nervana](https://www.intelnervana.com/technology/). [DSPs]([DSPs](https://en.wikipedia.org/wiki/Digital_signal_processor))
can also accelerate convolutions, a common operation in neural networks. 

### WebAssembly
WebAssembly is a new low-level assembly-like language with a compact binary format and 
near-native performance. Because programs written in C/C++ can be compiled directly to 
WebAssembly to run in browsers, WebAssembly addresses the developer pain in importing
models trained in modern frameworks. 

However, the current WebAssembly designs do have a few limitations when it comes 
to inferencing: 
1. WebAseembly doesn't yet have GPU support, a well-known performance accelerator for ML.
1. Modules running in the WebAssembly cannot run multi-threads yet. Though developers can
    split their programs to have separate modules and load each into separate Web Worker 
    to have multi-thread support, this is an unnecessary pain for developers. 
1. Because the platform doesn't yet have good support for matrix computation and other 
    needed mathematical operations, each website would still need to load their own math
    libraries. This could lead to an over-bloated browser with multiple libraries for 
    the browsers to optimize. 

## Native Platform Support 

Recently major native platforms have shipped APIs to support neural network: 

* iOS and MacOS shipped 
    [Basic Neural Network Subroutines (BNNS)](https://developer.apple.com/documentation/accelerate/bnns) 
    and the [MPS Graph API](http://machinethink.net/blog/ios-11-machine-learning-for-everyone/)
* Universal Windows Platform (UWP) shipped 
    [support for model evaluation on UWP platform via the Cognitive Toolkit](https://blogs.windows.com/buildingapps/2017/08/31/cognitive-toolkit-model-evaluation-uwp/)
* Android shipped 
    [Neural Network API](https://developer.android.com/ndk/guides/neuralnetworks/index.html)

Native platforms have long had optimized mathematical libraries that can be hardware accelerated. 
Some of the mathematical libraries have come from standardization such as [Basic Linear 
Algebra Subprograms (BLAS)]http://www.netlib.org/blas/#_history), which 
has [a variety of implementations](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Implementations) 
on different platforms.

Other than platform API support, development frameworks can also build their executables to run
with hardware accelerations. The frameworks have helped the developer ecosystem to grow, 
especially because *training* is so critical in ML development. Examples include: 
* Google's [TensorFlow Lite and Mobile](https://www.tensorflow.org/mobile/)
* Microsoft's [Windows ML for UWP](https://docs.microsoft.com/en-us/windows/uwp/machine-learning/) and [CNTK support for UWP](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Library-Evaluation-on-UWP)
* Apple's [CoreML Framework](https://developer.apple.com/documentation/coreml)
* Facebook's [Caffe2go](https://code.facebook.com/posts/196146247499076/delivering-real-time-ai-in-the-palm-of-your-hand/) and [PyTorch](http://pytorch.org/)


## Appendix: Related Research 

The design of an appropriate API surface for machine learning inference should incorporate 
learnings from research about optimizing machine learning models to run on devices with 
low computational power such as IoT devices. The section covers a few sample 
techniques for inspiration purpose: quantization, huffman coding, discretization, and sparse matrix.

### Quantization
Quantization refers to a group of techniques to convert high precision floating point 
numbers typically used in the training phase to low precision compact format numbers. 
Doing so allows us to reduce the file size and accelerate the computation. 
This technique is particularly useful for DNNs.

During training, programs typically compute in high precision floating point numbers. That is 
because the biggest challenge in training is to get the models to work and floating number 
is best at preserving accuracy. After all, training a neural network is essentially a process of
continuously tweaking the weights of the network until a satisfactory result is obtained. 
Plus developers usually have access to lot of GPUs during training and GPUs work 
very well with floating point numbers. Doing so would allow training to run a 
lot faster so to not waste development time.

During inference, a key challenge is now shrinking the file size. Converting numbers 
represented with 32 or 64 bits into 8 bits or fewer numbers shrinks the file size and memory 
throughput by four times. The same goes for caches and SIMD instructions. Because machine 
learning algorithms are particularly adept at canceling out noise, 
reducing precision often doesn’t lead to 
too much decrease in accuracy. Although low precision may not matter that much for GPUs, it can 
matter a lot for DSPs which are usually designed to operate with 8-bit numbers. Nowadays most 
computers including smartphones come with DSPs.

TensorFlow website has an 
[excellent explanation](https://www.tensorflow.org/performance/quantization) 
for how quantization works with neural networks.

### Discretization
[Discretization](https://en.wikipedia.org/wiki/Discretization) is the process to transfer 
continuous functions to discrete numbers. Some may argue discretization is part of 
quantization, so it comes with the above-mentioned benefits. But this section calls 
out discretization because
exploratory research suggests this could significantly improve power consumption. Both of these
papers are great reads on this topic: 
[Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or −1](https://arxiv.org/pdf/1602.02830.pdf) and 
[Ternary Neural Networks for Resource-Efficient AI Applications](https://arxiv.org/pdf/1609.00222.pdf). 

### Huffman Coding
[Huffman coding](https://www.geeksforgeeks.org/greedy-algorithms-set-3-huffman-coding/) is 
a commonly used compression algorithm that uses variable-length codeword to encode symbols. 
It's been applied in many areas such as video and audio codecs to further compress after other 
techniques have been applied. 

[Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) 
suggest Huffman coding can usually shrink network file 
size by about 20% to 30%. It can be used at the end of the optimization process to further
reduce its size. The de-compression process can also be accelerated by 
[DSPs](https://en.wikipedia.org/wiki/Digital_signal_processor). 

### Sparse Matrix
Most machine learning problems don’t involve a densely populated matrix. For example, on a map,
the cities are very densely populated, but most of the wilderness are scarcely populated.
Adopting sparse 
representation of those matrices and computation methods for sparse representation, such as 
[SparseBLAS](http://math.nist.gov/spblas/), can significantly reduce the size of the models. [A Survey of Sparse Representation:
Algorithms and Applications](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7102696) 
describes a variety of algorithms applicable to sparse representations of matrix. 


<br>
<br>
<br>
<br>
<br>
<br>
<a name="myfootnote1">1</a>: A application with its core functionality provided by a machine 
learning model. 

<br>

<a name="myfootnote2">2</a>: In the below paragraphs, models are used to describe all machine 
learning programs while networks are only used for artificial neural networks. 

<br>

<a name="myfootnote3">3</a>: Personally I am very impressed by the surge of growth of 
related libraries. 
I started researching this topic in Aug and could only find a handful of high quality
libraries. But when I re-visited the topic in Dec, I've found many more great frameworks. More
than that, developer conferences this year are filled with annoucement in this area, such as the 
Android Neural Network API which was still in speculation in Aug or the MPS Graph API which is
the evolution of the original BNNS API. What is encouraging to me is that people seem to 
shift from thinking that we should provide a strictly defined framework and a few basic models 
to developers to the idea that 
we should provide basic building blocks to developers and see what they will come up with. 

<br>

<a name="myfootnote4">4</a>: In machine learning research, performance can sometimes refer to 
    the accuracy of the models at achieving human-comparable result. In the context of 
    this explainer, performance strictly refers to processing speed. 

<br>

<a name="myfootnote5">5</a>: Admittedly the process is an overgeneralization because 
different developers will likely develop differently. For example, some developer may start
with a neural network already written in JS and try to optimize it for other things. 

<br>

<a name="myfootnote6">6</a>: Many factors contribute to this phenomenon. Here are the few main 
factors: 
1. During the training process, the main focus is on rapid iteration of the model and ease of 
    expressing the model. Compared to other major languages, python is easiest at expressing
    mathematical formula. New students to machine learning typically learn things in Python as
    well. 
2. It's until the past few years that JavaScript has decent support for raw buffers. 
    Majority of machine learning codes have to do with matrix manipulations. Using raw buffer 
    or at least objects with buffers to represent matrix is a much more memory saving option. 
3. There is a pre-conceived notion that JavaScript is slow. Abhishek Soni's 
    [Machine Learning with JavaScript](https://hackernoon.com/machine-learning-with-javascript-part-1-9b97f3ed4fe5) 
    clarifies this preconceived notion well. 
4. Because the majority of the industry has gone with the option of python + C, the cloud 
    platforms are optimized for that. Developers these days rarely train anything in local 
    environment but usually use cloud platforms such as Azure, AWS, or GCP. Though developers
    can load their libraries in the VMs and do things their own way, it is often a lot 
    easier just go with the default. 
5. Because of this trend of industry, there is a much larger community of developers working on
    machine learning in python. Developers get much more support using the default option. 

<br>

<a name="myfootnote7">7</a>: The module loaded in the WebAssembly VM cannot instantiate a new
    thread yet. But developers can create multiple 
    [Web Workers](https://html.spec.whatwg.org/multipage/workers.html#workers) and 
    load individual module inside it to run in multi-thread fashion. 

