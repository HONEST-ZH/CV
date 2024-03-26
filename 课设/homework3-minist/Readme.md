# 传统方法和深度学习方法对比  
### 传统方法
传统方法的原始训练数据是./data/0 到./data/9下的手写数字图片。 
传统方法的过程为：读取图像，图像预处理，倾斜矫正，计算HOG，训练SVM，测试SVM。
#### 读取图像
使用glob库的golbe方法，获得所有的图像的文件名。使用cv.imread读取这些图像。
#### 图像预处理
使用cv.cvtColor转变为灰度图、cv.resize调整大小
#### 倾斜矫正
使用cv.moments获得图像的矩。图像的矩用于判断在y轴上的倾斜程度。并根据倾斜度生成一个仿射矩阵实现倾斜的矫正。  
计算矩阵的矩使用了cv.moments方法，返回一个字典。其中：
* mu02：y坐标的二阶中心矩，描述了在y轴上的倾斜情况。
* mu11：x和y坐标的混合二阶中心矩。  
使用这两个数据可以计算倾斜度，倾斜度是mu11/mu02
#### 计算HOG
针对图像计算出图像的梯度分布直方图
#### 训练SVM
#### 测试SVM
训练的数据集规模太小。使用MINIST数据集生成一个图片数量大小为100的数据集。对比不同数据规模和是否使用倾斜矫正的准确率。  
可以观察到使用倾斜校正的SVM方法准确率高于不使用倾斜矫正的SVM方法。数据规模越大，准确率越高，使用倾斜校正的准确率增益也越小。
#### 传统方法的改进
## 深度学习方法
深度学习方法则从minist数据集进行训练和测试。  

# 输出结果
# 结论
# 依赖环境 
    anyio==4.3.0
    row==1.3.0
    asttokens==2.4.1
    async-lru==2.0.4
    attrs==23.2.0
    Babel==2.14.0
    beautifulsoup4==4.12.3      
    bleach==6.1.0
    certifi==2024.2.2
    cffi==1.16.0
    charset-normalizer==3.3.2   
    colorama==0.4.6
    comm==0.2.1
    contourpy==1.2.0
    cycler==0.12.1
    debugpy==1.8.1
    decorator==5.1.1
    defusedxml==0.7.1
    executing==2.0.1
    fastjsonschema==2.19.1
    filelock==3.9.0
    fonttools==4.49.0
    fqdn==1.5.1
    fsspec==2023.4.0
    h11==0.14.0
    httpcore==1.0.4
    httpx==0.27.0
    idna==3.6
    ipykernel==6.29.3
    ipython==8.22.1
    ipywidgets==8.1.2
    isoduration==20.11.0
    jedi==0.19.1
    Jinja2==3.1.2
    joblib==1.3.2
    json5==0.9.17
    jsonpointer==2.4
    jsonschema==4.21.1
    jsonschema-specifications==2023.12.1
    jupyter==1.0.0
    jupyter-console==6.6.3
    jupyter-events==0.9.0
    jupyter-lsp==2.2.3
    jupyter_client==8.6.0
    jupyter_core==5.7.1
    jupyter_server==2.12.5
    jupyter_server_terminals==0.5.2
    jupyterlab==4.1.2
    jupyterlab_pygments==0.3.0
    jupyterlab_server==2.25.3
    jupyterlab_widgets==3.0.10
    kiwisolver==1.4.5
    MarkupSafe==2.1.3
    matplotlib==3.8.3
    matplotlib-inline==0.1.6
    mistune==3.0.2
    mpmath==1.3.0
    nbclient==0.9.0
    nbconvert==7.16.1
    nbformat==5.9.2
    nest-asyncio==1.6.0
    networkx==3.2.1
    notebook==7.1.1
    notebook_shim==0.2.4
    numpy==1.26.3
    opencv-contrib-python==4.9.0.80
    opencv-python==4.9.0.80
    overrides==7.7.0
    packaging==23.2
    pandocfilters==1.5.1
    parso==0.8.3
    pillow==10.2.0
    platformdirs==4.2.0
    prometheus_client==0.20.0
    prompt-toolkit==3.0.43
    psutil==5.9.8
    pure-eval==0.2.2
    pycparser==2.21
    Pygments==2.17.2
    pyparsing==3.1.2
    python-dateutil==2.8.2
    python-json-logger==2.0.7
    pywin32==306
    pywinpty==2.0.13
    PyYAML==6.0.1
    pyzmq==25.1.2
    qtconsole==5.5.1
    QtPy==2.4.1
    referencing==0.33.0
    requests==2.31.0
    rfc3339-validator==0.1.4
    rfc3986-validator==0.1.1
    rpds-py==0.18.0
    scikit-learn==1.4.1.post1
    scipy==1.12.0
    Send2Trash==1.8.2
    six==1.16.0
    sniffio==1.3.1
    soupsieve==2.5
    stack-data==0.6.3
    sympy==1.12
    terminado==0.18.0
    threadpoolctl==3.4.0
    tinycss2==1.2.1
    torch==2.2.1+cu121
    torchaudio==2.2.1+cu121
    torchvision==0.17.1+cu121
    tornado==6.4
    traitlets==5.14.1
    types-python-dateutil==2.8.19.20240106
    typing_extensions==4.8.0
    uri-template==1.3.0
    urllib3==2.2.1
    wcwidth==0.2.13
    webcolors==1.13
    webencodings==0.5.1
    websocket-client==1.7.0
    widgetsnbextension==4.0.10

