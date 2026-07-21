# 30. Notebook Train MSVC Environment

날짜: 2026-07-16

## 원인

저장된 Train-cell log가 iteration 0에서 멈췄다. OSN-GS CUDA rasterizer preflight가 cl.exe를 찾지 못했기 때문이다. CUDA extension setup cell은 임시 build child에서만 vcvars64.bat를 활성화했지만, Train은 MSVC PATH, INCLUDE, LIB이 없는 별도 train.py subprocess를 시작했다.

## 수정

Train cell이 임시 cmd launcher를 작성하고 vcvars64.bat를 호출한 뒤 set으로 environment를 수집해 train.py subprocess environment에 합치도록 변경했다. launch 전에 where cl을 probe하며 activation이 불완전하면 원인을 지정한 error를 보고한다. 임시 cmd 방식은 성공했던 CUDA extension-build cell과 동일하다.

## 검증

notebook JSON과 Train-cell Python syntax가 compile된다. 새 helper를 local VS Build Tools installation에서 실행했고 non-empty INCLUDE/LIB과 함께 x64 MSVC path의 cl.exe를 찾았다.

## 다음 실행

kernel을 재시작했다면 CUDA extension setup cell을 다시 실행한 뒤 Train을 실행한다. training command 전에 Train MSVC environment가 출력돼야 한다.
