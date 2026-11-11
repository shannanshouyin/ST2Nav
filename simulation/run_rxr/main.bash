export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# 与原版ETP-R1一致：先读取iter_train.yaml，再用下列命令行参数覆盖运行资源。
# 模型、训练和消融开关仍以iter_train.yaml为准，避免在两个文件重复维护。
# 实验名读取yaml中的EXP_NAME，便于SFT/GRPO分开保存；资源覆盖保持原样。
flag_train="--run-type dagger
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 6
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "

flag_eval="--run-type eval
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 6
      IL.back_algo control
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "

flag_inference="--run-type inference
      --exp-config run_rxr/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "

mode=$1
master_port=$2

if [ "${mode}" = "eval" ] && [ -n "${3:-}" ]; then
      flag_eval="${flag_eval} EVAL.CKPT_PATH_DIR $3"
fi

case ${mode} in
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port ${master_port} run.py ${flag_train}
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port ${master_port} run.py ${flag_eval}
      ;;
      inference|infer)
      echo "###### inference mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port ${master_port} run.py ${flag_inference}
      ;;
      *)
      echo "Usage: CUDA_VISIBLE_DEVICES=0 bash run_rxr/main.bash {train|eval|inference} PORT [EVAL_CKPT]"
      exit 2
      ;;
esac

# CUDA_VISIBLE_DEVICES=0 bash run_rxr/main.bash train 2334
# CUDA_VISIBLE_DEVICES=0 bash run_rxr/main.bash eval 2334
