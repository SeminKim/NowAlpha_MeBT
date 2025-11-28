# README: NowAlpha_MeBT

이 프로젝트는 3D VQGAN과 트랜스포머를 사용한 강수량 예측 시스템입니다. 이 문서는 모델을 훈련하고 사용하는 전체 워크플로우를 안내합니다.

## 목차
1. [환경 설정](#환경-설정)
2. [3D VQGAN 훈련](#3d-vqgan-훈련)
3. [HDF5 파일 생성](#hdf5-파일-생성)
4. [트랜스포머 훈련](#트랜스포머-훈련)
5. [강수량 예측](#강수량-예측)

## 환경 설정

기본적으로 도커 이미지를 사용하여 환경을 설정합니다.
먼저 아래 명령어를 통해 도커 이미지를 설치합니다.
```bash
docker load -i [도커_이미지_파일명].tar
```
그런 다음 도커 컨테이너를 실행합니다. (run_docker.sh 참고)

혹 모듈 중 설치되지 않은 패키지가 있다면 아래 명령어 참고 부탁드립니다.
```bash
pip install -r requirements.txt
```

## 3D VQGAN 훈련

첫 번째 단계는 3D VQGAN 모델을 훈련하는 것입니다. 이 모델은 기상 데이터의 시공간적 특성을 효과적으로 인코딩합니다.

### hdf5 준비

create_hdf5_for_3d.py 사용.
- 깨진 파일이 있는지 사전 검사하는 목적.
- 모든 dbz파일에 대한 list가 저장된 all_dBZ.txt 파일 생성.
    - `find $(pwd) -name "*.bin.gz" >> all_dBZ.txt`
- h5py.File에 저장.
- 학습에 사용.

### 설정 파일

`configs/satellite_3dVQ/` 디렉토리에서 적절한 설정 파일을 선택하세요. 예를 들어:

```bash
# 설정 파일 살펴보기
ls configs/satellite_3dVQ/
```

다음은 `configs/satellite_3dVQ/magvit_base/magvit_dBZ_10min_16f_vq2_pGAN_0.01.yaml` 설정 파일의 예시입니다:

```yaml
model:
    target: tats.tats_vqgan.VQGAN
    params:
      load_path: {ckpt_path}
      mse_codebook: True
      image_channels: 16
      embedding_dim: 1024  # 256 in MAGVIT paper
      n_codes: 1024
      n_hiddens: 64
      downsample: [1, 16, 16]
      disc_channels: 64
      disc_layers: 5
      discriminator_iter_start: 1
      disc_loss_type: hinge
      image_gan_weight: 0.0
      video_gan_weight: 0.01
      f1_weight: 0.0
      l1_weight: 4.0
      gan_feat_weight: 0
      perceptual_weight: 0
      i3d_feat: False
      restart_thres: 1
      no_random_restart: True
      norm_type: group
      padding_type: reflect
      normalize_input: True
      mask_disc: False

data:
    vqgan: True
    batch_size: 1
    val_batch_size: 1
    num_workers: 8
    image_channels: 16
    data_interval: 10
    input_interval: 10
    before: 60
    output_interval: 10
    after: 100
    validation_use_also_train_set: False
    sort_val: True
    vtoken: False
    spatial_to_channel: [4, 4]
    pooling: [1024, 1024]
    hsr_data_path: 'datasets/all_dBZ.h5'
    gz_data_path: 'datasets/HSR_dBZ'

    train:
        include: []
        exclude: ['2020.*', '201703.*', '201705.*']
    test:
        include: ['2020.*']
        exclude: []

exp:
    base_lr: 0.0
    exact_lr: 1e-5
    cosine_lr: True
    warmup_steps: 1000
    gradient_clip_val: 1.0
```

3D VQGAN 모델 훈련은 이 설정 파일을 사용하여 진행됩니다. 이 설정은 MAGVIT 기반의 모델로, 1024 차원의 임베딩과 코드북을 사용하며, video_gan_weight가 0.01로 설정되어 있습니다.

### 모델 훈련

다음 명령어로 3D VQGAN 모델을 훈련합니다:
모델 학습 시 사용한 reference config는 아래와 같습니다.

1. configs/satellite_3dVQ/magvit_base/magvit_dBZ_10min_16f_vq2_1024.yaml (Discriminator 없이 먼저 recon_loss 수렴할 때까지 학습.)
2. configs/satellite_3dVQ/magvit_base/magvit_dBZ_10min_16f_vq2_pGAN_0.01.yaml (Discriminator 활성화 후 학습.)

```bash
python train_3dvq.py --base configs/satellite_3dVQ/magvit_base/magvit_dBZ_10min_16f_vq2_pGAN_0.01.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --default_root_dir [DIR] \
    --check_val_every_n_epoch=1 \
    --limit_val_batches 0.01 \
    --max_steps 2000000 \
    --accumulate_grad_batches 1 \
    --num_nodes 1 \
    --nproc_per_node=1 \
    --precision 32 \
    --gradient_clip_val 1.0
```

훈련이 완료되면 체크포인트 파일이 `KMA_3DVQ` 디렉토리 아래에 저장됩니다.

## HDF5 파일 생성

훈련된 3D VQGAN 모델을 사용하여 원시 데이터를 압축된 잠재 표현으로 변환하고 HDF5 파일로 저장합니다.

### HDF5 생성

HDF5 파일 생성을 위해 `eval_3dvq.py`를 사용합니다. 다음과 같이 두 가지 HDF5 파일을 생성해야 합니다:

1. **타겟 데이터를 위한 HDF5**: `magvit_dBZ_10min_6to6.yaml` 설정 파일을 사용하여 타겟 데이터의 잠재 표현을 생성합니다.
2. **컨텍스트 데이터를 위한 HDF5**: `magvit_dBZ_10min_replicate_6to6.yaml` 설정 파일을 사용하여 컨텍스트 데이터의 잠재 표현을 생성합니다.
3. 저장경로는 config 파일 내 save_dir에 지정된 경로에 저장됩니다.

다음 명령어로 HDF5 파일을 생성합니다:

```bash
# 타겟 데이터 HDF5 생성
python eval_3dvq.py \
    --base configs/satellite_3dVQ/finals/magvit_dBZ_10min_6to6.yaml \
    --ckpt_path [3DVQGAN_체크포인트_경로] \
    --gpus 0,1,2,3,4,5,6,7

# 컨텍스트 데이터 HDF5 생성
python eval_3dvq.py \
    --base configs/satellite_3dVQ/finals/magvit_dBZ_10min_replicate_6to6.yaml \
    --ckpt_path [3DVQGAN_체크포인트_경로] \
    --gpus 0,1,2,3,4,5,6,7
```

이 과정은 모든 훈련 및 검증 데이터를 처리하고 이를 HDF5 파일 형식으로 저장합니다.
파일들은 각 GPU별로 따로 저장됩니다. 이를 합치기 위해서 aggregate_h5s.ipynb를 사용합니다.

* f.close() 쉘까지 돌리시면 됩니다.


## 트랜스포머 훈련

다음 단계는 생성된 HDF5 파일을 사용하여 트랜스포머 모델을 훈련하는 것입니다. 이 모델은 3D VQGAN을 통해 인코딩된 잠재 표현으로부터 미래 프레임을 예측합니다.

### 설정 파일

`configs/satellite/` 디렉토리에서 적절한 설정 파일을 선택하세요. 예를 들어:

```bash
# 설정 파일 살펴보기
ls configs/satellite/
```

다음은 `configs/satellite/MeBT24_6to6hrs_droppath.yaml` 설정 파일의 예시입니다:

```yaml
model:
    target: tats.COMMIT_transformer.Net2NetTransformerVToken  # 모델 클래스 타겟
    params:
        load_dBZ: True  # dBZ 로드 여부
        unconditional: True  # 비조건부 모드 사용
        vocab_size: 1024  # 어휘 크기
        first_stage_vocab_size: 1024  # 첫 번째 단계 어휘 크기
        block_size: 18432  # 블록 크기
        n_layer: 24  # 트랜스포머 레이어 수
        n_head: 16  # 어텐션 헤드 수
        n_embd: 1024  # 임베딩 차원
        n_unmasked: 0  # 마스킹되지 않는 토큰 수
        embd_pdrop: 0.1  # 임베딩 드롭아웃 비율
        resid_pdrop: 0.1  # 잔차 드롭아웃 비율
        attn_pdrop: 0.1  # 어텐션 드롭아웃 비율
        path_pdrop: 0.3  # 경로 드롭아웃 비율
        sample_every_n_latent_frames: 0  # 잠재 프레임 샘플링 간격
        first_stage_key: video  # 첫 번째 단계 키
        interior_key: interior_video  # 내부 비디오 키
        cond_stage_key: label  # 조건부 단계 키
        t_prior: longest  # 시간 우선순위 전략
        decode:
            steps: 5  # 디코딩 스텝 수
            sampling_temperature: 1.0  # 샘플링 온도
            mask_temperature: 4.5  # 마스크 온도

    mask:
        target: tats.mask_sampler.CommitMaskGen  # 마스크 생성기 타겟
        params:
            iid: False  # IID 마스킹 사용 여부
            schedule: cosine  # 마스킹 스케줄
            max_token: 18432  # 최대 토큰 수
            method: 'mlm'  # 마스킹 방법
            shape: [72, 16, 16]  # 마스크 형태
            t_range: [0.0, 0.5]  # 시간 범위
            budget: 18432  # 마스킹 예산

    vqvae:
        target: tats.tats_vqgan.VQGAN_decoder  # VQGAN 디코더 타겟
        params:
          load_path: KMA_3DVQ/1ap18xwq/checkpoints/latest_checkpoint.ckpt  # VQGAN 체크포인트 경로
          mse_codebook: True  # MSE 코드북 사용 여부
          image_channels: 16  # 이미지 채널 수
          embedding_dim: 1024  # 임베딩 차원
          n_codes: 1024  # 코드북 크기
          n_embed: 1024  # 임베딩 수
          n_hiddens: 64  # 히든 레이어 크기
          downsample: [1, 16, 16]  # 다운샘플링 비율

data:
    load_dBZ: True  # dBZ 로드 여부
    vqgan: False  # VQGAN 데이터 모드 사용 여부
    batch_size: 2  # 배치 크기
    val_batch_size: 2  # 검증 배치 크기
    num_workers: 3  # 데이터 로더 워커 수
    image_channels: 16  # 이미지 채널 수
    COMMIT: True  # COMMIT 알고리즘 사용 여부
    data_interval: 10  # 데이터 간격(분)
    input_interval: 10  # 입력 간격(분)
    before: 360  # 과거 데이터 길이(분)
    output_interval: 10  # 출력 간격(분)
    after: 360  # 미래 예측 길이(분)
    use_time: True  # 시간 정보 사용 여부
    validation_use_also_train_set: False  # 훈련 세트도 검증에 사용할지 여부
    val_vtoken: True  # 검증에 토큰 사용 여부
    sort_val: True  # 검증 데이터 정렬 여부
    vtoken_train:
        hsr_data_path: 'datasets/2024_6to6/2024_3dvq_aggregated_chunk.h5'  # 훈련 데이터 경로
        hsr_interior_path: 'datasets/2024_6to6/2024_3dvq_replicate_aggregated_chunk.h5'  # 내부 데이터 경로
        include: []  # 포함할 데이터 패턴
        exclude: ['2020.*', '201703.*', '201705.*']  # 제외할 데이터 패턴

exp:
    accumulate_grad_batches: 3  # 그래디언트 누적 배치 수
    base_lr: 0.0  # 기본 학습률
    exact_lr: 3e-4  # 정확한 학습률
    cosine_lr: True  # 코사인 학습률 스케줄링 사용
    warmup_steps: 30000  # 웜업 스텝 수
```

### 모델 훈련

다음 명령어로 트랜스포머 모델을 훈련합니다:
* tmp 자리에는 아무 문자열이나 들어가도 됩니다. (사용안함)
```bash
bash scripts/train_config_log_gpus.sh configs/satellite/MeBT24_6to6hrs_curriculum.yaml tmp 0,1,2,3,4,5,6,7
```

## 강수량 예측

마지막 단계는 훈련된 모델을 사용하여 미래 강수량을 예측하는 것입니다.

### 예측 생성

`generate_vis_nc_v2.py` 스크립트를 사용하여 예측을 생성하고 시각화합니다:

```bash
python generate_vis_nc_v2.py \
    --config configs/3dVQ_transformer_base.yaml \
    --ckpt_path [트랜스포머_체크포인트_경로] \
    --exp [결과_디렉토리_이름] \
    --n_steps 5 \
    --sampling_temperature 1.0 \
    --mask_temperature 4.5 \
    --n_samples 1 \
    --after 360 \
    --batch_size 4 \
    --save_nc \
    --target_dates [타겟_날짜_파일.txt]
```

이 명령어는 다음 작업을 수행합니다:
- 지정된 날짜에 대한 강수량 예측 생성
- NetCDF 파일 형식으로 결과 저장
- 결과 시각화 (선택적)

## 결과 분석

결과는 다음 디렉토리에 저장됩니다:
- NetCDF 파일: `nc_results/[결과_디렉토리_이름]/`
- 시각화 결과: `vis_results/[결과_디렉토리_이름]/`
- 평가 지표: `nc_results/[결과_디렉토리_이름]/metrics/`


## Acknowledgement
This work was supported in part by the "HPC support" project funded by the Korea government.
