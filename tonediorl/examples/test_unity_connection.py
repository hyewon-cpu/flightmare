#!/usr/bin/env python3
"""
RL 학습하는거 말고 그냥 시뮬레이션 띄워보고 싶었음.
기본적으로 Unity를 띄우고, env를 만들어서 연결해줘야함.
주기적으로 step을 해줘야 rendering을 유지할 수 있음.

Step 1: Unity 연결 테스트
Unity 실행 파일을 먼저 실행한 후, 이 스크립트를 실행하세요.

Unity 실행:
    cd ~/projects/flightmare/flightrender/RPG_Flightmare
    ./RPG_Flightmare.x86_64
"""

import os
import io
import time
import numpy as np
from ruamel.yaml import YAML

from tonedio_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1


def main():
    print("=" * 60)
    print("Step 1: Unity 연결 테스트")
    print("=" * 60)
    print("\n먼저 Unity를 실행했는지 확인하세요:")
    print("  cd ~/projects/flightmare/flightrender/RPG_Flightmare")
    print("  ./RPG_Flightmare.x86_64\n")
    
    input("Unity가 실행되었으면 Enter를 누르세요...")
    
    # 1. Config 로드
    print("\n[1] Config 파일 로드 중...")
    yaml = YAML()
    cfg_path = os.path.join(os.environ["FLIGHTMARE_PATH"], "flightlib/configs/vec_env.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f)
    
    # 단일 환경으로 설정
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = "yes"  # Unity 렌더링 활성화
    
    # YAML 문자열로 변환
    stream = io.StringIO()
    yaml.dump(cfg, stream)
    cfg_yaml_str = stream.getvalue()
    
    print("✓ Config 로드 완료")
    
    # 2. Environment 생성
    print("\n[2] Environment 생성 중...")
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_yaml_str, False))
    print("✓ Environment 생성 완료")
    print(f"  - obs_dim: {env.num_obs}")
    print(f"  - act_dim: {env.num_acts}")
    
    # 3. Unity 연결
    print("\n[3] Unity 연결 시도 중...")
    try:
        env.connectUnity()
        print("✓ Unity 연결 성공!")
        
        # 4. 환경 리셋 (드론 초기화)
        print("\n[4] 환경 리셋 중...")
        obs = env.reset()
        print("✓ 환경 리셋 완료")
        print(f"  - 초기 위치: [{obs[0, 0]:.2f}, {obs[0, 1]:.2f}, {obs[0, 2]:.2f}]")
        
        # 5. 주기적으로 렌더링 업데이트 전송 (Unity 연결 유지)
        print("\n[5] Unity 렌더링 루프 시작...")
        print("  Unity 창에서 드론이 보이나요?")
        print("  (주기적으로 업데이트를 보내서 연결을 유지합니다)")
        print("\n  종료하려면 Ctrl+C를 누르세요...\n")
        
        frame_count = 0
        last_print_time = time.time()
        
        try:
            while True:
                # 제로 액션으로 step 호출 (드론은 그대로 유지, Unity 업데이트만 전송)
                zero_action = np.zeros((1, env.num_acts), dtype=np.float32)
                obs, reward, done, info = env.step(zero_action)
                
                frame_count += 1
                
                # done이 True면 C++에서 자동으로 reset이 호출됨
                if done[0]:
                    print(f"\n[Episode Terminated] 드론이 바닥에 떨어졌습니다!")
                    print(f"  C++ VecEnv가 자동으로 reset()을 호출했습니다.")
                    print(f"  새로운 초기 상태로 리셋되었습니다.\n")
                
                # 1초마다 상태 출력
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    pos = obs[0, 0:3]
                    done_str = " [TERMINATED]" if done[0] else ""
                    print(f"Frame {frame_count} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | Done: {done[0]}{done_str}")
                    last_print_time = current_time
                
                # 약간의 딜레이 (너무 빠르게 보내지 않기 위해)
                time.sleep(0.02)  # ~50Hz
                
        except KeyboardInterrupt:
            print("\n\n루프 종료")
        
    except Exception as e:
        print(f"✗ Unity 연결 실패: {e}")
        print("\n확인사항:")
        print("  1. Unity 실행 파일이 실행 중인가요?")
        print("  2. Unity가 완전히 로드되었나요?")
        return
    
    # 6. 정리
    print("\n[6] 연결 종료 중...")
    env.disconnectUnity()
    env.close()
    print("✓ 정리 완료")


if __name__ == "__main__":
    main()
