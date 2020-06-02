# Solutions to various errors ::: 각종 에러 해결 
Solutions to various errors ::: 에러 발생 시 대처 및 해결 방안

Errors
===

---
* keras or tensorflow import시 numpy버전 때문에 발생하는 에러 
  * type 관련 에러
  ```
  FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.

  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
  ```
  * 해결 방법
    * numpy 다운그레이드
    * 프롬프트에서 `pip install "numpy<1.17` -> 관리자 권한으로 실행

* locale 설정  
    * Tensorflow 돌리려는데 다음과 같은 에러가 발생.
    * locale.Error: unsupported locale setting
    ```
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8') 인 기존 코드를 다음과 같이 변경
    locale.setlocale(locale.LC_ALL, '')
    ```

---
* Information 성 불필요 메시지(messages) 미출력
    ``` 
    # 1. Info성 불필요 메시지 미출력을 위한 작업
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ```
    ```
    # 2. User 에러 미표시
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    ```

---
* GPU 사용시 설정 해줘야 할 것
  * 추천 방법
    ```
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    ```
  * 차선택
    ```
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    ```

---
* keras,tensorflow gpu version install : 재설치 및 세팅
  * tensorflow 설치 : `pip install --ignore-installed --upgrade tensorflow-gpu` or `pip install tensorflow-gpu`
    * tensorflow 확인 
      ```
      import tensorflow as tf
      hello = tf.constant('Hello, TensorFlow!')
      sess = tf.Session()
      print(sess.run(hello))
      ```
  * keras 설치 : `conda install -c anaconda keras-gpu` or `conda install keras-gpu`
    * keras 확인
      * `import keras`
  * Numpy 설치 : `pip install "numpy<1.17`
  * [선택] pytorch 설치 : `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
  * [선택] conda 업데이트 : `conda update -n base -c defaults conda`
  * [선택] pip 업데이트 : `python -m pip install --upgrade pip`
  * [선택] 파이썬 버전 다운그레이드 : `conda install python=3.6`
  * [선택] 가상환경 만들기 : `conda create -n koos_keras python=3.6`

  * 패키지 설치
    * [선택] opencv 설치 : `pip install opencv-python` -> `import cv2`

* 연구실 컴퓨터 GTX 2080 TI에 맞는 CUDA, Cudnn, tensorflow version
  * `CUDA` : 10.0
  * `Cudnn` : 7.4.x (7.4.1 추천)
  * `tensorflow-gpu` : 1.13.1
    * `pip install --upgrade tensorflow-gpu==1.13.1`
    * 버전 확인 : https://www.tensorflow.org/install/source_windows#tensorflow_1x=
  * numpy 오류 나기 때문에 재설치
    * `pip install "numpy<1.17"`
  * 참고 : https://hansonminlearning.tistory.com/7
    
* 현재 저장되어 있는 패키지 목록 추출 및 재설치
  * 패키지 추출 : `pip freeze > requirements.txt`
  * 패키지 설치 : `pip install -r requirements.txt`
  * requirements.text 버전 설치
    * 버전 이상 설치 : `idna>=2.8`
    * 2버전대의 아무 버전이나 설치 : `idna>=2.*`

* iopub 등의 쥬피터 노트북 메모리 부족 
  * error example
    ```
    using Plots; plotly()
    plot(real(sol[:,1]))
    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    ```
  * start jupyter notebook at terminal : `jupyter notebook --NotebookApp.iopub_data_rate_limit=2147483647`
  * Reference : https://github.com/JuliaLang/IJulia.jl/issues/528
