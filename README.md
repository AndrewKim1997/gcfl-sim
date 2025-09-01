```
gcfl-sim/
├─ README.md                    # 상단에 gcfl 링크, 사용/확장/가속/분산 안내
├─ LICENSE
├─ CITATION.cff
├─ CHANGELOG.md
├─ CONTRIBUTING.md              # 플러그인 규약/코드 스타일
├─ .gitignore                   # Python+Numba+CMake+결과물+Ray/Dask 로그
├─ .dockerignore
├─ pyproject.toml               # extras: "fast"(Numba/pybind11) 등
├─ environment.yml              # conda 대안(옵션)
├─ Makefile                     # build/run/test/benchmark 단축키(옵션)
├─ src/
│  └─ gcfl/                     # 패키지(임포트명은 gcfl)
│     ├─ __init__.py
│     ├─ rng.py                 # 결정적 서브스트림(seed→run/repeat/round/client)
│     ├─ types.py               # dataclass/TypedDict 스키마
│     ├─ params.py              # 설정 검증/변환
│     ├─ aggregates.py          # mean/median/k-trim/정렬가중
│     ├─ signals.py             # s_i 생성, 잡음 모형
│     ├─ mechanisms.py          # 보상/제재, 임계/예산
│     ├─ dynamics.py            # 라운드 상태 갱신
│     ├─ metrics.py             # M, PoG, PoC, ΔU 등
│     ├─ engine.py              # 메인 루프(signals→aggregate→mechanism→update→log)
│     ├─ io.py                  # CSV/Parquet 로깅, 메타데이터 기록
│     ├─ viz.py                 # 기본 시각화(옵션)
│     ├─ run.py                 # 단일 실험(Reference/Scale 공통 CLI)
│     └─ sweep.py               # 파라미터 스윕(단일/분산 백엔드)
├─ accel/
│  ├─ numba_kernels.py          # 선택: Numba JIT 커널
│  └─ cpp/
│     ├─ CMakeLists.txt
│     ├─ fast_kernels.cpp       # 선택: 핵심 루프 가속
│     └─ pybind_module.cpp      # pybind11 바인딩
├─ configs/
│  ├─ base.yaml                 # 엔진 기본값
│  ├─ profiles/
│  │  ├─ standard.yaml
│  │  ├─ large.yaml
│  │  └─ xl.yaml
│  └─ sweeps/
│     ├─ alpha_pi.yaml
│     └─ stress_boundary.yaml
├─ scripts/
│  ├─ quickstart.sh             # 바로 실행 예시
│  ├─ make_figs.py              # 데모용 플롯
│  ├─ benchmark.py              # 성능/스케일 벤치
│  └─ profile.sh                # cProfile/py-spy 등(옵션)
├─ results/
│  ├─ logs/.gitkeep
│  ├─ figures/.gitkeep
│  └─ cache/.gitkeep
├─ docker/
│  ├─ Dockerfile                # CPU 기본
│  ├─ Dockerfile.cuda           # 선택: CUDA
│  └─ docker-compose.yml        # 선택: Ray/Dask 분산
├─ tests/
│  ├─ test_aggregates.py        # 동률/절단 규칙 불변성
│  ├─ test_engine_determinism.py# 결정성/시드 재현
│  ├─ test_metrics.py           # 지표 검증
│  ├─ test_scale_vs_ref.py      # Scale↔Reference 오차 한계
│  └─ property/
│     └─ test_properties.py     # 속성 기반/경계 조건
└─ .github/
   └─ workflows/
      ├─ ci.yml                 # lint+unit+property+간단 스윕
      ├─ docker.yml             # 이미지 빌드/GHCR 푸시(옵션)
      └─ build_wheels.yml       # 배포용 휠 빌드(옵션)
```
