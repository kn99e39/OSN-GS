# 31. 통합 PowerShell CLI

날짜: 2026-07-16

## 작업

표준 packaging metadata와 osn-gs console script를 추가했다. top-level help는 train, benchmark, inspect-surface, stream-server를 노출한다. 각 command는 구현을 복사하지 않고 기존 entry point에 위임하므로 command별 help와 동작이 기준이 된다.

## 사용

repository virtual environment를 활성화한 뒤 osn-gs --help를 실행한다. 변경을 받은 뒤 local editable installation을 갱신하려면 다음 명령을 실행한다.

    .venv\Scripts\python.exe -m pip install -e . --no-deps

## 검증

설치된 console script에서 top-level help와 train, benchmark command help를 확인했다.
