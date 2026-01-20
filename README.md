# Flightmare

![Build Status](https://github.com/uzh-rpg/flightmare/workflows/CPP_CI/badge.svg) ![clang format](https://github.com/uzh-rpg/flightmare/workflows/clang_format/badge.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg) ![website]( https://img.shields.io/website-up-down-green-red/https/naereen.github.io.svg)

**Flightmare** is a flexible modular quadrotor simulator.
Flightmare is composed of two main components: a configurable rendering engine built on Unity and a flexible physics engine for dynamics simulation.
Those two components are totally decoupled and can run independently from each other. 
Flightmare comes with several desirable features: (i) a large multi-modal sensor suite, including an interface to extract the 3D point-cloud of the scene; (ii) an API for reinforcement learning which can simulate hundreds of quadrotors in parallel; and (iii) an integration with a virtual-reality headset for interaction with the simulated environment.
Flightmare can be used for various applications, including path-planning, reinforcement learning, visual-inertial odometry, deep learning, human-robot interaction, etc.

**[Website](https://uzh-rpg.github.io/flightmare/)** & 
**[Documentation](https://flightmare.readthedocs.io/)** 

[![IMAGE ALT TEXT HERE](./docs/flightmare_main.png)](https://youtu.be/m9Mx1BCNGFU)

## Installation
Installation instructions can be found in our [Wiki](https://github.com/uzh-rpg/flightmare/wiki).

### 리펙토링 후 설치 방법
- Prerequisites
    
    ```jsx
    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       libzmqpp-dev \
       libopencv-dev
    ```
    

- 가상환경 사용 추천
    
    ```jsx
    mamba create --name flightmare python=3.10
    mamba activate flightmare
    ```
    

- Flightmare 설치
    
    ```jsx
    git clone https://github.com/uzh-rpg/flightmare.git
    ```
    
- 환경 변수 설정
    
    ```bash
    echo "export FLIGHTMARE_PATH=~/Desktop/flightmare" >> ~/.bashrc
    source ~/.bashrc
    ```
    

- dependency 설치
    
    ⚠️ tensorflow 안써보려고 함
    
    ```bash
    cd flightmare/
    
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
     
    # install scikit
    pip install scikit-build
    
    pip install wandb
    pip install tensorboard
    pip install tqdm rich
    ```
    
- flightlib 설치
    
    ⚠️ [setup.py](http://setup.py) 바꿨다고 가정
    
    ```bash
    cd flightmare/flightlib
    # it first compile the flightlib and then install it as a python package.
    pip install .
    ```
    

→ 여기서 설치되는거 = numpy, stable-baselines3, gymnasium, ruamel.yaml


### RL 디렉토리

  ```bash
  conda activate ENVNAME
  cd /path/to/flightmare/flightrl
  pip install -e .
  cd examples
  python3 run_drone_control.py --train 0 --render 1
  ```

→ 여기서 설치되는거 = numpy, stable-baselines3, gymnasium, ruamel.yaml

- 이미지 랜더링 하려면 아까 위에서 flightrender 아래 추출한 파일 실행
    
    → flightmare UI 실행됨
    
    ```bash
    ./RPG_Flightmare.x86_64
    ```
    

###  실행

```bash
python run_drone_control.py --train 1 --render 0 --wandb 1 --wandb_project flightmare_ppo
```

  
## Updates
 *  17.11.2020 [Spotlight](https://youtu.be/8JyrjPLt8wo) Talk at CoRL 2020 
 *  04.09.2020 Release Flightmare

## Publication

If you use this code in a publication, please cite the following paper **[PDF](http://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)**

```
@inproceedings{song2020flightmare,
    title={Flightmare: A Flexible Quadrotor Simulator},
    author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
    booktitle={Conference on Robot Learning},
    year={2020}
}
```

## License
This project is released under the MIT License. Please review the [License file](LICENSE) for more details.
