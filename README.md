# Solutions to various errors & TIPS ::: 각종 에러 해결 
Solutions to various errors ::: 에러 발생 시 대처 및 해결 방안

TIP
===

### VS Code
* vscode 초기 세팅
  * 확장 tool 설치
    * `Markdowns All in One`
    * `Markdowns+Math`
    * `VS Code Jupyter Notebook`
    * `Markdown PDF`
    * `Korean Language Pack for Visual Studio Code`
  * 테마
    * `Monokai` 또는 `Visual Studio Dark`

* vs code latex(수식) 포함하여 pdf 저장하는 방법
  * `.md` 맨 아랫줄에 아래 코드 삽입하고 저장한 뒤 pdf로 저장하면 된다.
    ```
    <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>
    ```
* vs code : markdown 색상 입력
  * `<span style="color:Aqua ">AlphaZero</span>`
  * color 색상 표 참고 : https://css-tricks.com/snippets/css/named-colors-and-hex-equivalents/

Errors
===

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
* keras 백엔드(backend)가 theano로 설정되어 있을 때 해결 방법
  * `C:\Users\(사용자이름)\.keras\keras.json`에 들어가서 Backend를 theano에서 tensorflow로 바꿔주면 된다.
  * https://3months.tistory.com/138

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

* anaconda, keras, tensorflow 설치
  * 1) Anaconda 설치
  * 2) Anaconda 설치 후 Anaconda Prompt 실행
  * 3) 가상환경 생성 : [  conda create n 가상환경이름 python=3.6   ] 입력
  * 4) 가상환경 접속 : [  activate 가상환경이름   ]
  * 5) tensorflow 설치 : [ pip install tensorflow==1.14 ] or [ GPU version  pip install tensorflow-gpu==1.14 ]
  * 6) keras 설치 : [  pip install keras==2.2.5  ]
  * 7) 주피터노트북 설치 : [  pip install jupyter ]
  * 8) numpy 다운그레이드 : [  pip install numpy ==1.16.1  ] 
  * 9) pillow, opencv 설치 : [  conda install pillow opencv   ]
  * 10) 주피터노트북 실행 :[ jupyter notebook ]

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

  * `vs code 필수 라이브러리 설치 목록`
    * `Markdown+Math` : 수식 입력
    * `Prettify JSON` : json 파일 정렬 라이브러리
    * `Markdown PDF` : PDF 파일로 만들기
    * `Korean Language Pack for visual ---` : 한국어 패치

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

* 파이썬 64비트와 32비트 함께 사용하기(파이썬 32비트 설치)
  * conda 프롬프트에서 아래 코드 실행
  * `conda create -n py36_32`
  * `conda activate py36_32`
  * `conda config --env --set subdir win-32`
  * `conda install python=3.6`
  * `conda info` 실행 후 플랫폼에서 win-32인지 확인하기
  * `conda deactivate` 실행 후 base의 플랫폼이 win-64인지 확인하기
  * 출처 : https://m.blog.naver.com/haanoon/221814660104
  * 참고 하면 좋음 : 
  
* 대신증권 api 이용
  * 핸드폰 어플 `CYBOS Touch` 설치 후 비대면계좌 개설
  * 비대면 계좌 개설 후 증권용 공인인증서 등록
  * CYBOS Plus에 로그인
  * `http://blog.naver.com/PostView.nhn?blogId=hjinha2&logNo=221185064559` 여기 참조
  * python 32비트 설치 후 파이참 interpreter에서 파이썬 32비트 경로 지정
  * win32com : `conda install pywin32`
  * `DB Browser for SQLite` 설치 후 저장한 db확인
  * 참고 : https://excelsior-cjh.tistory.com/105

# 우분투(ubuntu 16.04) 환경 세팅
  * 1.anaconda3 설치
    * 설치 url : `https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/` 
    * ex) Anaconda3-2020.02-Linux-x86_64.sh
    * commend에서 `bash Anaconda3-2020.02-Linux-x86_64.sh` 실행
  * 2.가상환경 만들기
    * `conda create -n (가상환경이름) python=(파이썬버전)`
    * ex) `conda create -n koos_detect python=3.6`
    * 이후 `source ~/.bashrc` 실행하면 설치 완료
  * 3.주피터노트북 설치
    * 설치 : `pip install jupyter --user`
    * 설치 확인 : `jupyter notebook`
    * 기본 설정파일 생성 : `jupyter-notebook --generate-config`
    * 원격 연결시 사용할 비밀번호 설정하기
      * 커멘드 창에서 `ipython` 입력 후 아래 입력
      * `[1] from notebook.auth import security` 실행
      * ## 구버전의 경우 import security가 아닌 import password인 경우도 있다.
      * `[2] security.passwd()`
      * 패스워드 입력창이 나오는데 원격연결로 접속시 사용할 비밀번호를 입력 & 확인한다.
      * 그러면 아래 그림과 같이 Out: 'sha1:~~~' 과 같은 문자열이 출력되는데 전체를 복사한다.
      * ipython에서 `exit()`를 입력하시면 기존의 터미널창으로 돌아갈 수 있다.
    * VI에디터로 설정파일 수정하기
      * jupyter 위에서 생성한 환경설정 파일은 리눅스의 홈 디렉토리아래 `.jupyter`라는 폴더 내부에 생성된다. 환경설정 파일이 존재하는 위치로 이동한 후 vi에디터를 이용해 수정한다.
      * `cd ~` 디렉토리 이동
      * `cd .jupyter`
      * `vi jupyter_notebook_config.py`
      * 참고 : vi 에디터 단축키 -> `i`(입력(insert)모드로 전환), `dd`(커서가 위치한 줄 삭제), `:wq`(저장 및 종료), `:/(찾고싶은 내용)`(검색), ESC(모드 빠져 나오기)
      * 아래 사이트 참고하면 됨.
      * 참고 사이트 : http://blog.naver.com/PostView.nhn?blogId=skyshin0304&logNo=221587513170&parentCategoryNo=&categoryNo=31&viewDate=&isShowPopularPosts=true&from=search
    * open-cv imshow 안 될 때 아래 커멘드 입력
      ```
      sudo apt-get -y install libgtk2.0-dev
      sudo apt-get -y install pkg-config
      conda remove opencv
      conda update conda
      conda install --channel menpo opencv
      pip install opencv-contrib-python
      ```
