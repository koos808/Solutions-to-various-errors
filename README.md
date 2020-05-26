# Solutions to various errors ::: 각종 에러 해결 
Solutions to various errors ::: 에러 발생 시 대처 및 해결 방안

* keras or tensorflow import시 numpy버전 때문에 발생하는 에러 
  * type 관련 에러
  ```
  FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.

  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
  ```
  * 해결 방법
    * numpy 다운그레이드
    * 프롬프트에서 `pip install "numpy<1.17` -> 관리자 권한으로 실행
* Information 성 불필요 메시지(messages) 미출력
    ``` 
    # Info성 불필요 메시지 미출력을 위한 작업
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ```
    