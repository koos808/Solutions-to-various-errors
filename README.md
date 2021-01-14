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

* VS code terminal :: Powershell에서 Command Prompt로 바꾸는 방법
  * 기본으로는 Windows PowerShell이라고 저장되어 있으니 해당 Terminal을 Command Pormpot로 바꾼다.
  * `Ctrl+Shift+P`(팔레트 선택) -> `Shell` 입력 -> `Command Prompt` 선택 ->  "Ctrl + `" 누르고 휴지통 버튼 눌러서 없앤다.
  * 이후 "Ctrl + `"를 눌러서 Terminal 다시 실행시키면 (base)로 되어 있는 터미널이 보임.

Errors
===

* 주피터노트북 Tab, shortcut 안될 때 해결하는 가장 빠른 방법 (TAB completion)
  * `%config Completer.use_jedi = False` 실행시키고 사용하기

* 주피터노트북 download as pdf 할때 생기는 에러
  * 에러명
  ```
  500 : Internal Server Error
  The error was:

  nbconvert failed: xelatex not found on PATH, if you have not installed xelatex you may need to do so. Find further instructions at https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex.
  ```
  * 해결방법
    * `https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex` 접속 후 windows버전 **MikTex** 설치하기
    * 쥬피터노트북 껐다 키기 끝.

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

* sklearn 에러
  * 설치해도 계속해서 `ModuleNotFoundError: No module named 'sklearn'` 에러가 발생하는 경우
  * 주피터 노트북 실행해서 아래 코드 실행시키면 된다.
  ```
  import sys
  !{sys.executable} -m pip install sklearn
  ```

* pydot, graphviz 설치 순서
  ```
  pip install pyparsing
  pip install graphviz
  pip install pydot
  conda install graphviz
  => 이후 kerneral restart
  ```
  * keras model summary plot 그리기
    * `keras.utils.plot_model(model, "concate_model.png", show_shapes=True)`
---
* Information 성 불필요 메시지(messages) 미출력
    ``` 
    # 1. Info성 불필요 메시지 미출력을 위한 작업
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ```
    ```
    # 2. User 에러 미표시 무시
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # 3. WARNING:tensorflow:From C:\Python\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version. 에러 무시하는 방법(아래 코드 입력)
    
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
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
* tensorflow - GPU 에러
  * `AttributeError: module 'tensorflow.python.framework.ops' has no attribute '_TensorLike'` 과 같은 에러
  * 해결책 tensorflow 1.13.1 버전으로 재설치하기 

* 주피터 노트북으로 localhost 계속 열어서 사용하다보면 kernel restarting 에러가 발생한다.
  * `jupyter notebook list` 커멘드로 현재 열려있는 localhost 확인하기.
  * `jupyter notebook stop 8888` -> 하나씩 종료시키기
  * runtime 경로 : `C:\Users\koos\AppData\Roaming\jupyter\runtime`

* 주피터 노트북 버전 확인하는 방법
  ```
  #주피터 노트북에서 파이썬 버전 확인하는 법
  import sys
  print(sys.version)
  ```

* Anaconda error
  * Conda process error : `Multiple Errors Encountered.` 에러
  * 해결책 : 아나콘다 네비게이터 관리자 권한으로 열기

* anaconda, keras, tensorflow 설치
  * 1) Anaconda 설치
  * 2) Anaconda 설치 후 Anaconda Prompt 실행
  * 3) 가상환경 생성 : [  conda create n 가상환경이름 python=3.6   ] 입력
  * 4) 가상환경 접속 : [  activate 가상환경이름   ]
  * 5) tensorflow 설치 : [ pip install tensorflow==1.14 ] or [ GPU version  pip install tensorflow-gpu==1.14 ]
  * 6) keras 설치 : [  pip install keras==2.2.5  ]
  * 7) 주피터노트북 설치 : [  pip install jupyter ]
  * 8) numpy 다운그레이드 : [  pip install numpy==1.16.1  ] 
  * 9) pillow, opencv 설치 : [  conda install pillow opencv   ]
  * 10) 주피터노트북 실행 :[ jupyter notebook ]

---
* keras,tensorflow gpu version install : 재설치 및 세팅
  * `conda create --name YOUR_ENV_NAME python=3.6` 
  * tensorflow 설치 : `pip install tensorflow-gpu==1.13.1`(내가 사용하는 코드) or `pip install --ignore-installed --upgrade tensorflow-gpu` or `pip install tensorflow-gpu` or ` conda install -c anaconda tensorflow-gpu`
  * keras 설치 : `conda install keras-gpu` or `conda install -c anaconda keras-gpu`
    * keras 확인
      * `import keras`
  * Numpy 설치 : `pip install "numpy<1.17"`
  * [선택] pytorch 설치 : `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
  * [선택] conda 업데이트 : `conda update -n base -c defaults conda`
  * [선택] pip 업데이트 : `python -m pip install --upgrade pip`
  * [선택] 파이썬 버전 다운그레이드 : `conda install python=3.6`
  * [선택] 가상환경 만들기 : `conda create -n koos_keras python=3.6`

  * 패키지 설치
    * [선택] 각종 패키치 설치 : `pip install jupyter pandas matplotlib sklearn opencv-python`
    * [선택] opencv 설치 : `pip install opencv-python` -> `import cv2`
    * [선택] 주피터 노트북 커널 설정 : `pip install ipykernel`
      * Jupyter notebook에 현재 가상환경을 추가 : `python -m ipykernel install --user --name myvenv --display-name "PythonHome_p36"`
      * 위와 동일 `python -m ipykernel install --user --name [virtualEnv] --display-name "[displayKenrelName]"`
    * [선택] 주피터 노트북 nbextension 설치
      * `pip install jupyter_contrib_nbextensions`
      * 주피터 노트북에서 보일 수 있도록 등록 : `jupyter contrib nbextension install --user`
      * 주피터 노트북 재실행
      * 내가 주로 사용하는 기능
        * `Table of Contents`, `Variable Inspector`, `Nbextenxions dashboard tab`, `Nbextensions edit menu item`
        * `Codefolding`, `ExecuteTime`, `contrib_nbextensions_belp_item`, 
    * [선택] 주피터 노트북에서 실시간 memory 사용량 모니터링 방법
      * `pip install nbresuse`
      * 바로 안보이면 아래 코드 실행
      * `jupyter serverextension enable --py nbresuse --sys-prefix`
      * `jupyter nbextension install --py nbresuse --sys-prefix`
      * `jupyter nbextension enable --py nbresuse --sys-prefix`
      * 그래도 안되면 아래 코드 실행
      * `jupyter serverextension enable --py nbresuse`
      * 주피터 노트북 재실행

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
    * 
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

# 우분투(ubuntu 18.04) 환경 세팅
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
    * [선택] 설치 확인 : `jupyter notebook`
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
      * `cd ~/.jupyter`
      * `vi jupyter_notebook_config.py`
      * vi 실행 창에서 "i" 버튼 누르고 제일 상단에 하단 코드 작성(괄호 제외)
      * 
        ```
        c = get_config()
        c.JupyterApp.config_file_name = 'juyter_notebook_config.py'
        c.NotebookApp.allow_origin = '*' (접속 허용 ip – 본인 ip 아니면 * (전체 허용))
        c.NotebookApp.ip = 'xxx.xx.xxx((서버 ip)' 
        c.NotebookApp.open_browser = False # False이면 jupyter notebook 실행시 창이 아닌 url이 나온다.
        c.NotebookApp.password = u'아까 복사했던 그 sha1 ~~~~~ 여기에 복사하기'
        ```
      * esc 버튼 누르고 :wq 입력 후 엔터
      * [참고] : vi 에디터 단축키 -> `i`(입력(insert)모드로 전환), `dd`(커서가 위치한 줄 삭제), `:wq`(저장 및 종료), `:/(찾고싶은 내용)`(검색), ESC(모드 빠져 나오기)
      * 참고 사이트 : http://blog.naver.com/PostView.nhn?blogId=skyshin0304&logNo=221587513170&parentCategoryNo=&categoryNo=31&viewDate=&isShowPopularPosts=true&from=search
  * 4.각종 패키지 설치
    * tensorflow 설치 : `pip install tensorflow-gpu`
    * keras 설치 : `pip install keras`
    * 주피터 노트북 포함 각종 패키지 설치
      * `pip install jupyter pandas matplotlib sklearn opencv-python`
    * [선택] : `pip install keras_applications`
  * 5.[선택]쥬피터 노트북(Jupyter notebook) 테마 변경하기
    * thema 패키지 설치 : `pip install jupyterthemes`
    * 내 테마 설정 : `jt -t onedork -T -N -kl -f roboto -fs 11 -tfs 11 -nfs 13 -tfs 13 -ofs 10 -cellw 80% -lineh 170 -cursc r -cursw 6`
  * 6.[선택] 주피터 노트북 nbextension 설치
    * `pip install jupyter_contrib_nbextensions`
    * 주피터 노트북에서 보일 수 있도록 등록 : `jupyter contrib nbextension install --user`
    * 주피터 노트북 재실행
    * 내가 주로 사용하는 기능
      * `Table of Contents`, `Variable Inspector`, `Nbextenxions dashboard tab`, `Nbextensions edit menu item`
      * `Codefolding`, `ExecuteTime`, `contrib_nbextensions_belp_item`, 
  * 7.[선택] 주피터 노트북에서 실시간 memory 사용량 모니터링 방법
    * `pip install nbresuse`
    * 바로 안보이면 아래 코드 실행
    * `jupyter serverextension enable --py nbresuse --sys-prefix`
    * `jupyter nbextension install --py nbresuse --sys-prefix`
    * `jupyter nbextension enable --py nbresuse --sys-prefix`
    * 그래도 안되면 아래 코드 실행
    * `jupyter serverextension enable --py nbresuse`
    * 주피터 노트북 재실행


* open-cv imshow 안 될 때 아래 커멘드 입력
  ```
  sudo apt-get -y install libgtk2.0-dev
  sudo apt-get -y install pkg-config
  conda remove opencv
  conda update conda
  conda install --channel menpo opencv
  pip install opencv-contrib-python
  ```

* 서버에서 서버(서버간) 폴더, 파일 복사하기
  * sch 이용
  * instar 폴더를 B서버에 koos 폴더 안에 복사 
    * `scp -r /home/koos/instar remoteID@remoteIP주소:/home1/koos`
  * instar 파일을 B서버에 koos 폴더 안에 복사 
    * `scp -r /home/koos/instar/test.txt remoteID@remoteIP주소:/home1/koos/`
  * 조건 : 예를 들어 "A"서버에서 "B"서버로 폴더나 파일을 이동하려면 "B" 서버에 "A"서버 IP를 방화벽 접속 가능하도록 권한을 부여해야 한다.
  * 반대도 마찬가지로 파일 이동가능하도록 방화벽 권한을 주어야 한다.

# rtx 3090 setting
  * OS : windows10
  * VGA : RTX 3090
  * VGA Driver : 456.43
  * Cuda 설치 : `cuda_11.0.3_451.82_win10`(cuda 11.0 업데이트된 최신버전)
  * Cudnn 설치 : `cudnn-11.0-windows-x64-v8.0.3.33`(cudnn)
  * Anaconda 설치
  * Conda update : `conda update -n base conda`
  * 가상환경 생성 : `conda create -n env_name python=3.8` -> `conda activate env_name`
  * tensorflow-gpu 설치 : `pip install tf-nightly-gpu==2.5.0.dev20201102`
  * 각종 패키지 설치 : `conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch`
