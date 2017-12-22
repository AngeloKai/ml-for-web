# Explainer:  API Sets for Machine Learning on the Web 

With the recent breakthroughs in deep learning and related technologies, Machine Learning (ML) algorithms has drastically improved in terms of accuracy, application, performance etc. While 
typically thought of as a technology only applicable to server technologies, the 
inferencing process of machine learning models can run on device as well. Development of a 
machine learning application<sup>[1](#myfootnote1)</sup> usually involves two stages: training and inference: 

* The developer first *train* the model by first creating a skeleton framework and then iterating the model with large dataset 
* The developer then port the model to production environment so that it can *infer* insight from user input 

Though training typically takes place in the cloud because it requires large amount of data and computing power, inference can take place in the cloud or on the device. Running inference on the 
device has a number of appealing properties, such as performance boost 
due to [edge computing](https://en.wikipedia.org/wiki/Edge_computing), resistance toward poor or no network, and security/privacy protection, etc. 

Although platforms for native applications have all shipped APIs to support machine learning 
inference on device, similiar functionality has been missing on the web platform. The explainer 
recommends an approach to address this missing functionality by providing an improved WebAssembly
package, a WebML (Web Machine Learning) API with a pre-defined set of mathmatical functions 
that the platform can optimize for, and a WebNN (Web Neural Network) API that provides high level
abstraction to run neural networks efficiently. Other approaches are also welcomes. After all, 
explainer is meant for sparking conversations rather than pinpointing a clear solution. 

Supporting ML inferences can not only supercharge existing applications but also unlock new scenarios (see [use cases](#use-cases)). For instance, with the help of 
[Service Worker](https://github.com/w3c/ServiceWorker), applications can translate between 
languages with poor or no network. By inferring the user’s emotions 
based on user’s input (be it text, image, or video), developers can build a rich 
interactive experience. Applications on new frontiers such as Mixed Reaility can become much 
"smarter."

Developers have also shown strong interests in deploying inferences in 
web applications as evidenced by the growing number of machine learning libraries that can run in 
browsers.<sup>[2](#myfootnote2)</sup> See [here](https://github.com/AngeloKai/js-ml-libraries) for a short list of the libraries or frameworks. [Synaptic.js](http://caza.la/synaptic/#/) and [webdnn](https://mil-tokyo.github.io/webdnn/) are examples of the impressive works developers have done
to enable this scenario. 

However, though the libraries and frameworks have helped lower the barrier of development, 
developers continue facing a 
[painful development process along with bottlenecks](#developer-pain) 
because of limitations with the current platform. Developers would run into issues such as 
porting model from C to JS, large model size, performance<sup>[6](#myfootnote6)</sup>, 
memory overflow etc. Four existing 
standard efforts helped address the pain but none of them is a complete solution: 

1. [__High Level JS APIs Built On Machine Learning Technology.__](#APIs-Built-On-Machine-Learning-Technologies)

    A number of high level JavaScript APIs are built on machine learning technologies, such as  
    the [Web Speech API](https://dvcs.w3.org/hg/speech-api/raw-file/tip/webspeechapi.html), 
    [Web Authentication API](https://w3c.github.io/webauthn/), 
    [Shape Detection API](https://github.com/WICG/shape-detection-api). 
    
    However, these APIs only address a very specific scenarios instead of offering 
    [extensibility](https://extensiblewebmanifesto.org/) to developers.
    For example, developers may want to detect more than the 
    [three basic types of shapes](https://github.com/WICG/shape-detection-api#overview) offered 
    by the Shape Detection API. Additionally, it is incredibly challenging to define 
    interoperability standards for these high level APIs because the browsers are building them with with different machine learning models that have different accuracy rate. And it
    will only be more challenging with the growing complexity of the models. 

1. [__WebGL.__](#WebGL) 

    The [WebGL](https://www.khronos.org/webgl/) API is the most commonly API used in the top libraries because developers can create wrapper functions around shader object to run 
    matrix computations in GPUs. However, the wrappers aren't 
    developer-friendly and usually very taxing on memory. Functions that directly do matrix 
    computation would make developers' life much easier and much simpler for the platform to 
    optimize.
    
    It is true that the next version of WebGL (WebGL 3.0) could include supporting
    machine learning inferences into its charter. However, it may be difficult because inferencing and drawing graphics impose different requirements to matrix computations. For example, 
    the former only needs low level of precision whereas latter most often requires high level of
    precision. Some machine learning computations are also not entirely around matrix 
    manipulations, such as fast fourier transform or huffman coding. 

1. [__WebGPU.__](#WebGPU)

    The [WebGPU](https://webkit.org/wp-content/uploads/webgpu-api-proposal.html#api) API is 
    in development to expose modern GPU features, though the initial API proposal based on 
    [Metal](https://developer.apple.com/documentation/metalperformanceshaders) 
    appeared to be geared toward graphics rendering but rather than matrix computations. 
    
    Contrary to popular beliefs, chips other than GPUs can be much more effective at accelerating
    ML models. More and more companies have been producing 
    [ASIC (Application-Specific Integrated Circuit)(https://en.wikipedia.org/wiki/Application-specific_integrated_circuit)
    chips for inferencing. Modern artificial neural networks also often employ convolution
    that [DSPs (Digital Signal Processors)](https://en.wikipedia.org/wiki/Digital_signal_processor)
    are designed to accelerate. 

1. [__WebAssembly.__](#WebAssembly)

    WebAssembly addresses the developer pain of porting binary package of trained ML
    models to the web platform. However, it doesn't address the performance issue because it 
    doesn't compile to GPU, the primary performance driver for ML (not to mention other chips). . 

    Assuming WebAssembly does compile down to GPU (a feature in consideration as seen in 
    [WebGPU Future Features](https://github.com/WebAssembly/design/blob/master/FutureFeatures.md)), the immediate issue of performance would be addressed but we would still face the issue of large model size. Each model would have to load its own math library. The platform would also likely
    face a fragmented ecosystem with many libraries to optimize for. We could 
    potentially learn from our experience and the type of libraries loaded to develop an API 
    that all ML models should call into. The platform could then optimize toward this API and 
    ease the developer pain of needing to understand differences in low level stacks. 
    Finding this standard set could start from existing standards such as 
    [BLAS](http://www.netlib.org/blas/).

Looking at the four existing efforts and how web platforms supported graphics (WebAssembly + WebGL),
the best approach forward appeared to be WebAssembly + an API of optimized mathmatical functions (tentatively called
WebML). Some may argue that this approach would still leave us with the issue of large model size because of large frameworks and we could reduce it down further by developing an API with a higher level of abstraction that models typical neural networks. After all, neural networks have been the 
forefront of machine learning innovations for the last few years. This "neural network" API could 
have existing structure for developers to load weights in and an easy way for nodes of the network
to communicate, such as [SharedBuffer](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer).

This approach would roughly match how native platforms supported them. Native platforms have shipped solutions to help close the gaps: 
* support traditional ML algorithms such as decision tree learning or Bayesian algorithms by providing optimized linear algebra libraries and other optimized mathmatical functions 
* support the Deep Neural Networks approach (center of attention in the recent AI boom) by shipping dedicated DNNs API. If developers find the DNNs API not complex enough to handle their special case, they can also use the optimize linear algebra libraries. 

The above approach would be promising but other suggestions are welcomed. After all, this explainer 
is just meant to help spark conversations around ML on the web to ease developer pain. Please feel free to submit PRs to correct me or add additional points. 


 
## Use Cases
Despite the long history of machine learning research and applications, I think it is safe to say we are still uncovering the countless applications ML. Below illustrate some example use cases developers may use machine learning in front-end applications. The sample use cases are based on inspirations from existing demos and production sites/apps. 

### Text Translation 
A web application may wish to translate text from one language to another offline to be network resilient. For example, the Google Translate service trained a neural network to translate between languages and ported the network to its mobile app. The mobile app can be used offline, though translation may be better online. 

### Offline Recommendation Engine
A web application built with Service Work to be network resistant may wish to build its recommendation engine offline. For example, a site serving images/GIFs/video as content may wish to serve users smart content feed with content cached with Service Worker. Or a productivity application with many different features like Office may want to provide Help when the user is looking to know which feature they should use. 

### Object Detection from Images/Videos
A web application may wish to detect objects from images or videos. For example, [Baidu](https://en.wikipedia.org/wiki/Baidu) built convolutional neural networks (CNNs) into its mobile app so that the app can detect the primary object in the live camera feed and search related merchandise based on the result ([Mobile Deep Learning framework by Baidu](https://github.com/baidu/mobile-deep-learning)).

In addition to generic object detection, a developer may also want to tweak his/her specific object detection to hone in on certain areas. For instance, if the developer anticipates the users would show them certain types of images, they could train the models with only those images to get better accuracy:

* An application may train a objection detection model with only credit card number images to auto detect those numbers from live camera feed. Doing so reduces friction with entering credit card number and increases conversion rate from basic users to premium users. 
* An application for streaming/uploading videos may wish to perform live check of the camera feed to ensure the user isn’t showing obscene content for law compliance purpose.
* An application can train a model with images of skin cancer and regular skin to help doctors detect skin cancer ([Dermatologist-level classification of skin cancer with deep neural networks](http://www.nature.com/nature/journal/v542/n7639/full/nature21056.html?foxtrotcallback=true)).

An application may also wish to let the front-end code only to identify the primary objects in an image and leave the task of classification to the back-end. Object detection roughly includes two sub-tasks: localization and classification. The former refers to the task of finding the pixel mask of the primary object in an image. The latter refers to the task of classifying which category an image belongs. The latter typically includes a much larger program while the former should usually be small enough to fit in client-side code. In the above skin cancer recognition example, the application may want to let the front-end code to identify irregularities (e.g. a mole) and leave the task of classifying what kind of skin cancer the irregularities signal to the back-end.

### Risk Analysis 
A web application may wish to deploy a small-scale risk analysis model to determine whether the transaction should be pre-approved and leave the final decision to the full-scale risk models on the backend. Quick pre-approval improves user experience while reducing cost for running the model.

### Rich Interactive Experience
A web application may wish to build "smarter" user interaction models. For example, the application can guess the user’s emotion based on user input and proactively make recommendations. For example, a social media site may wish to detect user’s emotion when user’s typing the post and recommend the right emoji to use. If the user wishes to post picture alongside the post, the application can also recommend appropriately based on the post.

### Mixed Reality Experience 
A web application built for mixed reality platforms may wish to leverage machine learning to anticipate user intention and provide assistance as needed. For example, when a worker is learning how to operate a new machine, the application could show how to operate each component as the worker gazes at each component. To do so, the app developer will need a objection detection model that is tuned to detect the components because they aren't included in typical image detection databases like [ImageNet](http://www.image-net.org/).  

## Developer Pain 

Developers face significant pain when trying a machine learning models into their front-end 
codebase today. The below walks through a typical development process<sup>[3](#myfootnote3)</sup> and illustrates specific pain points: 

1. Assuming the developer already has a trained model, the developers first face the pain 
    of large model size. Most models trained with frameworks such as [Tensorflow](https://www.tensorflow.org) can have file size that ranges from a few hundred Mb to 
    more than one GB. Some frameworks optimizing for mobile experience shrink the model size
    down to around 20 Mb by applying techniques such as [quantizatio](#quantization).
    Unfortunately, the process isn't as simple as described. The techniques do run the risk
    of reducing accuracy so developers have to keep double checking throughout the optimization
    process. 

1. After the developer combat the model size challenge, they have to face the next challenge of 
    language choice. The most well-known machine learning frameworks today don't have the option 
    of training models in JavaScript. The typical option is writing the model in python 
    , which the frameworks can compile down to executable and sometimes C.<sup>[4](#myfootnote4)</sup> Developers can either transpile the model to JS or 
    re-compile it to work with WebAssembly. The former option means increased file size 
    but easier way to improve performance. Developers can easily re-write the model to be 
    multi-threaded, a feature not yet available in WebAssembly.<sup>[5](#myfootnote5)</sup> 
    Developers can also more easily accelerate the models with GPU level acceleration 
    with existing libraries of wrapper functions of WebGL. The latter would mean consistent
    file size but much harder route to improve performance, though there is a silver lining. 
    For computations that can hardly be accelerated by GPU, such as Fast Fourier Transform,
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

Fortunately our colleagues working on other platforms such as IoT, native apps, or cloud computing,
have been working on addressing similiar issues so we will have some great advisors along the 
journey. 


## Native Platform Support 

Recently major native platforms have shipped APIs to support neural network: 

* iOS and MacOS shipped 
    [Basic Neural Network Subroutines (BNNS)](https://developer.apple.com/documentation/accelerate/bnns) and the 
    [MPS Graph API](http://machinethink.net/blog/ios-11-machine-learning-for-everyone/)
* UWP platform shipped 
    [support for model evaluation on UWP platform via the Cognitive Toolkit](https://blogs.windows.com/buildingapps/2017/08/31/cognitive-toolkit-model-evaluation-uwp/)
* Android shipped 
    [Neural Network API](https://developer.android.com/ndk/guides/neuralnetworks/index.html)

Native platforms have long had optimized mathmatical libraires that can be hardware accelerated. 
Some of the mathmatical libraries have come from standardization such as [Basic Linear 
Algebra Subprograms (BLAS)]http://www.netlib.org/blas/#_history), which 
has [a variety of implementations](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Implementations) in different 
platforms.

Other than platform API support, development frameworks can also build their executables to run
with hardware accelerations. The frameworks have really helped the developer ecosystem to grow, 
especially because *training* is so critical in ML development. Examples include: 
* Google's [TensorFlow Lite and Mobile](https://www.tensorflow.org/mobile/)
* Microsoft's [CNTK support for UWP](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Library-Evaluation-on-UWP)
* Apple's [CoreML Framework](https://developer.apple.com/documentation/coreml)
* Facebook's [Caffe2go](https://code.facebook.com/posts/196146247499076/delivering-real-time-ai-in-the-palm-of-your-hand/) [PyTorch](http://pytorch.org/)



## Existing Standards 

With these challenges in mind, we look at whether any of the existing standard APIs can resolve the challenges or be amended to address them: 

### APIs Built On Machine Learning Technologies
In the past few years, we have added support for a few new APIs that relies on machine learning technologies:

* The [Web Speech API](https://dvcs.w3.org/hg/speech-api/raw-file/tip/webspeechapi.html) enables developers to easily convert text content to speech and speech content to text. Both features are possible because of advancements we made in the natural language processing field, a sub-field of machine learning. 

* The [Web Authentication API](https://w3c.github.io/webauthn/) enables web developers to authenticate users with strong authenticators, such as fingerprint scanners, facial recognition systems, usb tokens etc. Biometric authenticators all employ machine learning technologies one way or another. 

* The [Shape Detection API](https://github.com/WICG/shape-detection-api), a recent addition to the Web Incubator CG (WICG), allow developers to detect faces, barcodes, and text in live or still images. This API is a good example of 
a feature that uses hardware acceleration ([Image Signal Processors (ISPs)](https://en.wikipedia.org/wiki/Image_processor)) to run machine learning operations.  

* The WebRTC (Web Real-Time-Communication API) also offers face detection functionality
    depending on platform implementation. 

One of the common motivations behind building the above APIs is the underlying machine models are computationaly expensive to run. However, it is unscalable to continue adding APIs to the platform for the reason of computational cost. There should be a generic solution that can bring down the computational cost of doing machine learning on the web platform.

### WebGL
The WebGL API was designed to render 3D and 2D graphic content and make use of GPUs behind the scene when necessary. Given that most of graphic processing relies on matrix computation, web developers have developed [libraries that wrap around WebGL to accelerate matrix computation](https://github.com/AngeloKai/js-ml-libraries). 

However, such libraries are not developer-friendly and often very taxing on memory. Take the example of this [matrix multiplication method](https://github.com/waylonflinn/weblas/blob/master/index.js#L59). During the computation, method instantiated two RGBA texel array, tranposed one of the arrays, created 2 input textures and 1 output texture, activated the shader, binded input texture, set shader parameters, binded output texture, and finally called drawElements to calculate the matrix. After the calculation, it also has to unbind all the textures. A simple matrix multiplication should only need to instantiate one new matrix in memory instead of five (two arrays and three textures). 

Although next generation of WebGL API can include more support for direct mathmatical computation, one can argue that this goal is not aligned with the charter of an API that was designed for drawing graphics. Besides, the next version of WebGL (WebGL 3.0) are still far away given that Chrome and Firefox has just implemented the support for the 2.0 version earlier this year.

### WebAssembly
WebAssembly is a new low-level assembly-like language with a compact binary format and near-native performance. Programs written in C/C++ can be compiled directly to this format to run on the web. On the browsers, WebAssembly programs run in a sandbox that can be used alongside JavaScript.

As previously stated, systemic support for Machine Learning programs should aim for allowing programs to have the least memory needed, provide most performance support, and preferably ease developer pain in importing their trained model to the web. Mainstream machine learning frameworks usually can produce models in C++ format. Given the above three goals, WebAssembly seems like a fitting solution for ML.

However, the current WebAseembly design do have a few shortcomings when it comes to being applied to ML. First of all, WebAseembly does not have GPU support, a well-known performance accelerator for ML. Second, Bottlenecks brought by network conditions are often motivations behind doing ML computation on the client. Common matrix functions can be large in size. Because WebAssembly is running on a blank slate, the developers have to load related libraries by themselves. If the libraries are built into the platform, much less speed requirement is needed. For example, developers would have to define their own matrix/format data type. This is also not consistent with
the promise of web platform, a platform that just "writes once and works everywhere." 

### WebGPU
[WebGPU](https://webkit.org/wp-content/uploads/webgpu-api-proposal.html#api) API is a new incubating API that aims at exposing modern GPU features. Its initial API set is a derivation from the Metal language. Prototype for the API has landed in WebKit.

Although the API aims at exposing low-level GPU functionalities, its initial API set is primarily geared toward graphics rendering and not direct mathmatical computation. Research has also shown that while GPU accelerates computing, specialized chips can be designed in ways that make them much better at machine learning computing. For example, quantization, a common technique to shrink number to less-than-32-bit representation, has proven to be an efficient technique to reduce the size of programs. 

Companies have produced chips designed for machine learning for personal devices instead of using GPUs, such as Movidius' (an Intel company) Myriad VPU, the IBM’s TrueNorth chips, or Intel’s NervanaIf the aim of the WebGPU API is to expose interface for the modern GPU, it would not be suitable for the machine learning field.


## Appendix: Related Research 

The design of an appropriate API surface for machine learning inference should incorporate learnings from research about optimizing machine learning models to run on devices with low computational power such as IoT devices. The section covers a few sample techniques for inspiration purpose: quantization, huffman coding, discretization, and sparse matrix.
A common theme among the techniques is they are all trade-offs between accuracy and other qualities.

### Quantization
Quantization refers to a group of techniques ot convert high precision floating point numbers typically used in the training phase to low precision compact format numbers. Doing so allows us to reduce the file size and accelerate the computation. This technique is particularly useful for DNNs.
During the training stage, programs typically compute in high precision floating point numbers. That is because the biggest challenge in training is to get the models to work and floating number is best at preserving accuracy. After all tasks like training neural network is essentially keep tweaking the weights of the network until a satifatory result is obtained. Plus developers usually have access to lot of GPUs during training and GPUs work very well with floating point numbers. Doing so would allow training to run a lot faster so to not waste development time.
During the inference, the main challenge becomes the shrinking the file size. As it turns out, converting 32 bit numbers into 8 bit numbers shrinks the file size and memory throughput by four times. The same goes for caches and SIMD instructions. Because many machine learning algorithms are now well-equipped to handle statistical noise, reducing precision often doesn’t lead to too much decrease in accuracy. Although low precision may not matter that much for GPUs, it can matter a lot for DSPs which are usually designed to operate with 8 bit numbers. Nowadays most computers including smartphones come with DSPs.

### Huffman Coding
Huffman coding is a commonly used compression alogrithm that uses variable-length codeward to encode symbols. Studies suggest Huffman coding can usually shrink network file size by about 20% to 30%. The technique can be used after quantization to reduce size. DSPs are really good here. 

### Discretization
Discretization is the process to transfer continious functions to to discrete numbers. Some may argue quantization is part of discretization. One thing to call out about this technique is that this really helps decrease power consumption.

### Sparse Matrix
Most machine learning problems don’t involve a densely populated matrix. Adopting sparse matrix data structures and specifical numerical methods for those data structures can significantly reduce the size of the memory.

### Fast Fourier Transform 
Luckily DSPs are really good at convolution like operations. 


<br>
<br>
<br>
<a name="myfootnote1">1</a>: In the below paragraphs, models are used to describe all machine 
learning programs while networks are only used for artificial neural networks. 

<a name="myfootnote2">2</a>: Personally I am very impressed by the rate of growth in related 
quality libraries are produced etc. 
libraries. I started researching this topic in Aug and could only find a handful of qualities 
libraries. But when I re-visited the topic in Dec, I've found many more great frameworks.

<a name="myfootnote3">3</a>: Admittedly the process is an overgeneralization because 
different developers will likely develop differently. For example, some developer may start
with a neural network already written in JS and try to optimize it for other things. 

<a name="myfootnote4">4</a>: Many factors contribute to this phenomone. Here are the few main 
factors: 
1. During the training process, the main focus is on rapid iteration of the model and ease of 
    expressing the model. Compared to other major languages, python is easiest at expressing
    mathmatical formula. New students to machine learning typically learn things in Python as
    well. 
2. It's until the past few years that JavaScript has decent support for raw buffers. 
    Majroity of machine learning codes have to do with matrix manipulations. Using raw buffer 
    or at least objects with buffers to represent matrix is a much more memory saving option. 
3. There is a pre-conceived notion that JavaScript is slow. Abhishek Soni's 
    [Machine Learning with JavaScript](https://hackernoon.com/machine-learning-with-javascript-part-1-9b97f3ed4fe5) clarifies
    this preconceived notion really well. 
4. Because the majority of the industry has gone with the option of python + C, the cloud 
    platforms are optimized for that. Developers these days rarely train anything in local 
    environment but usually use cloud platforms such as Azure, AWS, or GCP. Though developers
    can load their libraries in the VMs and do things their own way, it is often a lot 
    easier just go with the default. 
5. Because of this trend of industry, there is a much larger community of developers working on
    machine learning in python. Developers get much more support using the default option. 

<a name="myfootnote5">5</a>: The module loaded in the WebAssembly VM cannot instantiate a new
    thread yet. But developers can create multiple [Web Workers](https://html.spec.whatwg.org/multipage/workers.html#workers) and load individual module
    inside it to run in multi-thread fashion. 

<a name="myfootnote6">6</a>: In machine learning research, performance can sometimes refer to 
    the accuracy of the models at achieving human-comparable result. In the context of 
    this explainer, performance strictly refers to process speed. 