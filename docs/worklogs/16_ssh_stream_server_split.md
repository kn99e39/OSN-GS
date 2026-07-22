# 16. SSH 스트림 서버 분리

## 수행 내용

- TorchOSNGSTrainer에서 trainer-owned WebSocket server 경로를 제거했다.
- training streaming은 --stream_url을 통한 WebSocket client 경로로 유지했다.
- scripts/start_trainer_stream.ps1을 독립 local stream server launcher로 교체했다.
- osn_gs/interop/trainer_ws_server.py를 training client의 snapshot JSON을 받고 renderer client에게 broadcast하는 loopback-only WebSocket stream server로 재구성했다.
- WebRenderer/README.txt에 SSH local port-forward workflow를 반영했다.

## 결과

streaming 경로가 다시 training과 분리됐다.

    OSN-GS train.py / notebook --stream_url ws://127.0.0.1:8080
      -> trainer host의 local stream server
      -> SSH local port forwarding
      -> renderer browser ws://localhost:8080

training loop는 더 이상 WebSocket server를 열거나 소유하지 않는다. training을 종료해도 stream server는 종료되지 않는다.

## 평가

snapshot JSON payload format은 변하지 않았다. browser ping message에는 stream server가 pong을 보내고, snapshot message는 연결된 다른 모든 client에게 relay된다.

## 남은 위험

- 독립 stream server는 활성 Python environment에 websockets package가 필요하다.
- server는 의도적으로 127.0.0.1에만 bind한다. 원격 renderer 접근에는 SSH port forwarding이 필요하다.

## 2026-07-15 노트북 인터럽트 정리

- notebook Train cell은 이제 _run_monitored_process() 안에서 KeyboardInterrupt를 잡아 active train.py subprocess를 종료한 뒤 interrupt를 다시 전달한다.
- 먼저 process.terminate()를 호출하고 최대 10초 기다리며, 종료되지 않으면 process.kill()로 전환한다.
- 이 cleanup은 training subprocess에만 영향을 준다. scripts/start_trainer_stream.ps1로 시작한 standalone stream server는 별도 process이므로 자신의 terminal에서 Ctrl+C로 종료해야 한다.
