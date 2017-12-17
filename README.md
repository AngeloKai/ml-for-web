# Explainer:  API Sets for Machine Learning on the Web 

With the recent breakthroughs in deep learning and related technologies, the performance of Machine Learning (ML) algorithms has significantly improved. While typically thought of a technology that can only be applied to server technologies, machine learning technology can run on device as well. Developing a machine learning model usually involves two stages: training and inference. In the first stage, the developer decides on a skeleton model and feed large dataset to the model in repeated iterations to *train* the model. Then the model would then be ported to production environment to infer insights based on real time incoming data. While training is typically performed on the cloud, Inference can occur in the cloud or on the device. Performing inference on the device has a number of appealing properties, such as performance boost due to [edge computing](https://en.wikipedia.org/wiki/Edge_computing), resistance toward poor or no network, and security/privacy protection, etc. 

Although platforms for native applications have all shipped APIs to support machine learning inference on device, similiar functionality has been missing on the web platform. Supporting it can not only supercharge existing applications but also unlock new scenarios (see [use cases](#use-cases)). For example, with the help of service worker, developers can have their text translation application to be available offline. By inferring the user’s emotions based on user’s input (be it text, image, or video), developers can build a rich emotional experience. Applications on new frontiers such as Mixed Reaility can become much "smarter."

Developers have also shown strong interests in the method of deploying machine learning models in web applications as evidenced by the growing number of machine learning libraries that can run in browsers. See [here](https://github.com/AngeloKai/js-ml-libraries) for a short list of the libraries or frameworks. [Synaptic.js](http://caza.la/synaptic/#/) and [webdnn](https://mil-tokyo.github.io/webdnn/) are probably most impressive ones in the list. 

But today when web developers want to run machine learning models in their web applications, they face bottlenecks in terms of memory, performance, and power consumptions. The above-mentioned libraries typically use WebGL to help them improve performance and occasionly use WebAssembly or WebGPU. This is because, broadly speaking, the most performance consuming and most frequent operations in ML inference are matrix computations. Developers cleverly leverage them to accelerate the performance. 

However, because of a lack of comprehensive look at how to best support machine learning inference in front-end applications, there are still gaps left behind. Native platforms have shipped solutions to help close the gaps: supporting traditional ML algorithms such as decision tree learning or Bayesian algorithms by providing optimized linear algebra libraries and supporting the Deep Neural Networks approach (center of attention in the recent AI boom) by shipping dedicated DNNs API. Similiar solutions can potentially be developed for the web platform. Regardless, having a comprehensive solution would really help reduce developer pain and encourage these types of applications to grow. 

P.S. The explainer is just a mean to help spark conversations around ML on the Web. Please feel free to submit PRs to correct me or add additional points. 
 
## Use Cases
Despite the long history of machine learning research and applications, I think it is safe to say we are still uncovering the countless applications ML. Below illustrate some example use cases developers may use machine learning in front-end applications. The sample use cases are based on inspirations from existing demos and production sites/apps. A common theme to draw from the use cases is that ML on the Web will not only supercharge existing functionalities but also unlock new scenarios. 

### Offline Recommendation Engine
A web application built with Service Work to be network resistant may wish to build its recommendation engine offline. For example, a site serving images/GIFs/video as content may wish to serve users smart content feed with content cached with Service Worker. Or a productivity application with many different features like Office may want to provide Help when the user is looking to know which feature they should use. 

### Text Translation 
A web application may wish to translate text from one language to another offline to be network resilient. For example, the Google Translate service trained a neural network to translate between languages and ported the network to its mobile app. The mobile app can be used offline, though translation may be better online. 

### Object Detection from Images/Videos
A web application may wish to detect objects from images or videos. For example, Baidu built convolutional neural networks (CNNs) into its mobile app so that the app can detect the primary object in the live camera feed and search related merchandise based on the result (Baidu deep learning framework).

The problem of object detection involves two main sub-problems: localization and classification. The former refers to the problem of finding the location of the primary object in an image. The latter refers to the problem of classifying which category an image belongs. Solving the two problems together let us detect what kind of objects are in an image. 

In addition to generic object detection, a developer may also want to tweak his/her specific object detection to hone in on certain areas. For example, an application may wish to train their model to focus on a few classes of objects for the sake of more accurate detection. For example, an application may want to automatically detect the user's credit card number with live camera feed. Here the developer only wants a model trained with credit card images. There are other examples. Web application for streaming/uploading videos may wish to perform live check of the camera feed to ensure the user isn’t showing obscene content for law compliance purpose. Or a web application may allow users to diagnose whether they likely have skin cancers themselves with live camera feed (Esteve et al., 2017).

An application may also wish to let the front-end code only to identify the primary objects in an image and leave the task of classification to the back-end. Object detection roughly includes two sub-tasks: localization and classification. The former refers to the task of finding the pixel mask of the primary object in an image. The latter refers to the task of classifying which category an image belongs. The latter typically includes a much larger program while the former should usually be small enough to fit in client-side code. For example, in the above skin cancer recognition example, the application may want to let the front-end code to identify the mole and leave the task of classifying what kind of skin cancer the mole indicates to the back-end.

### Risk Analysis 
A web application may wish to deploy a small-scale risk analysis model to determine whether the transaction should be pre-approved and leave the final decision to the full-scale risk models on the backend. Quick pre-approval improves user experience while reducing cost for running the model.

### Rich Interactive Experience
A web application may wish to build "smarter" user interaction models. For example, the application can guess the user’s emotion based on user input and proactively make recommendations. For example, a social media site may wish to detect user’s emotion when user’s typing the post and recommend the right emoji to use. If the user wishes to post picture alongside the post, the application can also recommend appropriately based on the post.

### Mixed Reality Experience 
A web application built for mixed reality platforms may wish to leverage machine learning to anticipate user intention and provide assistance as needed. For example, when a worker is learning how to operate a new machine, the application could show how to operate each component as the worker gazes at each component. To do so, the app developer will need a objection detection model that is tuned to detect the components because they aren't included in typical image detection databases like [ImageNet](http://www.image-net.org/).  

## Native Platform Support 

### APIs
As described above, native platforms have supported machine learning through linear algebras libraries and all recently shipped neural network APIs. iOS and MacOS shipped Basic Neural Network Subroutines (BNNS) and updated Accelerate Framework for Core ML. The Universal Windows Platform (UWP) has added support for CNTK. Android is also said to release a Deep Neural Network API soon.

Unlike native platforms, web platforms do have its own unique challenges. First of all, many machine learning models can have a big file size. Although Service Worker has really helped close the gap for offline support, storing big file size can still be a challenge. Second, an app with machine learning functionalities can be published to only certain devices that match the performance requirement. The web platform cannot do that and developer will have to figure out a graceful fallback. The platform may have to provide some sort of device capability detection mechanism. 

### Frameworks 

Platform and developers have also built extensive frameworks on top of these APIs for mobile scenarios. Examples include Facebook’s [PyTorch](http://pytorch.org/) and Caffe2go, Google’s TensorFlow Lite, Apple’s CoreML Framework, and CNTK’s support for UWP.

We include frameworks because of two reasons: 1) unlike traditional programming, machine learning development place significant emphasis on the training phase. The final API should make it easy to port the trained models. 2) These frameworks usually are the first one adopting the APIs. 

## Challenges

File size, memory, performance, and power consumption. And we need something to detect device capability.  

## Existing Standards 

With those challenges, we look at what standards are available today: 

### APIs Relied on Machine Learning Technologies
In the past few years, we have added support for a few new APIs that relies on machine learning technologies. The Web Speech API enables developers to easily convert text content to speech and speech content to text. Both features are possible because of advancements we made in the natural language processing field, a sub-field of machine learning. The Web Authentication API enables web developers to authenticate users with strong authenticators, such as fingerprint scanners, facial recognition systems etc. Biometric authenticators all employ machine learning technologies one way or another. The Shape Detection API, a recent addition to the Web Incubator CG, allow developers to detect faces, barcodes, and text in live or still images. Object detection technologies are often based on research in machine learning, which in turn furthered research in Image Signal Processors (ISPs).
One of the common motivations behind building the above APIs are the machine models are computationaly expensive. Yet it is unscalable to continue adding APIs to the platform for the reason of computational cost. There should be a generic solution that can bring down the computational cost of doing machine learning on the web platform.

### WebGL
The WebGL API was designed to render 3D and 2D graphic content and make use of GPUs behind the scene when necessary. Given that most of graphic processing relies on matrix computation, web developers have developed libraries that wrap around WebGL to accelerate matrix computation. However, as I illustrate below, such libraries are not developer-friendly and often very taxing on memory. Take the example of this matrix multiplication method. The method has to first instantiates two RGBA texel array, tranpose one of the arrays, create 2 input textures and 1 output texture, activate the shader, bind input texture, set shader parameters, bind output texture, and finally call drawElements to calculate the matrix. After the calculation, it also has to unbind all the textures. A simple matrix multiplication should only need to instantiate one new matrix in memory instead of five ( two arrays and three textures).
Although next generation of WebGL API can include more support for direct mathmatical computation, one can argue that this goal is not aligned with the charter of an API that was designed for drawing graphics. In addition, the next version of WebGL (WebGL 3.0) are still far away given that Chrome and Firefox has just implemented the support for the 2.0 version earlier this year.

### Web Assembly
WebAssembly is a new low-level assembly-like language with a compact binary format and near-native performance. Programs written in C/C++ can be compiled directly to this format to run on the web. On the browsers, WebAssembly programs run in a sandbox that can be used alongside JavaScript.
As previously stated, systemic support for Machine Learning programs should aim for allowing programs to have the least memory needed, provide most performance support, and preferably ease developer pain in importing their trained model to the web. Mainstream machine learning frameworks usually can produce models in C++ format. Given the above three goals, WebAssembly seems like a fitting solution for ML.
However, the current WebAseembly design do have a few shortcomings when it comes to being applied to ML. First of all, WebAseembly does not have GPU support, a well-known performance accelerator for ML. Second, WebAssembly lacks support for running in WebWorkers. Because ML models can take up to several hundred megabytes and unpredictable, developers should be discouraged from running the models in the UI thread. Third, Bottlenecks brought by network conditions are often motivations behind doing ML computation on the client. Common matrix functions can be large in size. Because WebAssembly is running on a blank slate, the developers have to load related libraries by themselves. If the libraries are built into the platform, much less speed requirement is needed. For example, developers would have to define their own matrix/format data type.

### WebGPU
WebGPU API is a new incubating API that aims at exposing modern GPU features. Its initial API set is a derivation from the Metal language. Prototype for the API has landed in WebKit.
Although the API aims at exposing low-level GPU functionalities, its initial API set is primarily geared toward graphics rendering and not direct mathmatical computation. Research has also shown that while GPU accelerates computing, specialized chips can be designed in ways that make them much better at machine learning computing. For example, quantization, a common technique to shrink number to less-than-32-bit representation, has proven to be an efficient technique to reduce the size of programs. Companies have produced chips designed for machine learning for personal devices instead of using GPUs, such as Movidius' (an Intel company) Myriad VPU, the IBM’s TrueNorth chips, or Intel’s NervanaIf the aim of the WebGPU API is to expose interface for the modern GPU, it would not be suitable for the machine learning field.


## Appendix: Related Research 

The design of an appropriate API surface for machine learning inference should incorporate learnings from research about optimizing machine learning models to run on devices with low computational power such as IoT devices. The section covers a few sample techniques for inspiration purpose: quantization, huffman coding, discretization, and sparse matrix.
A common theme among the techniques is they are all trade-offs between accuracy and other qualities.

### Quantization
Quantization refers to a group of techniques ot convert high precision floating point numbers typically used in the training phase to low precision compact format numbers. Doing so allows us to reduce the file size and accelerate the computation. This technique is particularly useful for DNNs.
During the training stage, programs typically compute in high precision floating point numbers. That is because the biggest challenge in training is to get the models to work and floating number is best at preserving accuracy. After all tasks like training neural network is essentially keep tweaking the weights of the network until a satifatory result is obtained. Plus developers usually have access to lot of GPUs during training and GPUs work very well with floating point numbers. Doing so would allow training to run a lot faster so to not waste development time.
During the inference, the main challenge becomes the shrinking the file size. As it turns out, converting 32 bit numbers into 8 bit numbers shrinks the file size and memory throughput by four times. The same goes for caches and SIMD instructions. Because many machine learning algorithms are now well-equipped to handle statistical noise, reducing precision often doesn’t lead to too much decrease in accuracy. Although low precision may not matter that much for GPUs, it can matter a lot for DSPs which are usually designed to operate with 8 bit numbers. Nowadays most computers including smartphones come with DSPs.

### Huffman Coding
Huffman coding is a commonly used compression alogrithm that uses variable-length codeward to encode symbols. Studies suggest Huffman coding can usually shrink network file size by about 20% to 30%. The technique can be used after quantization to reduce size.

### Discretization
Discretization is the process to transfer continious functions to to discrete numbers. Some may argue quantization is part of discretization. One thing to call out about this technique is that this really helps decrease power consumption.

### Sparse Matrix
Most machine learning problems don’t involve a densely populated matrix. Adopting sparse matrix data structures and specifical numerical methods for those data structures can significantly reduce the size of the memory.

