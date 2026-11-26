# Commands (copy and paste on the cluster)

Run everything from inside the `NeuralP` folder, on a **GPU node**.

## 1. Get the code

First time:
```bash
git clone https://github.com/shruthiBala02/NeuralP.git
cd NeuralP
```

Every later time:
```bash
cd NeuralP
git pull
```

## 2. Create the environment (first time only)

```bash
# if your cluster uses modules, first:  module load python/3.11
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo 'export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8' >> .venv/bin/activate
```

## 3. Enter the environment (every new terminal or job)

```bash
source .venv/bin/activate
```

## 4. Check the GPU

```bash
nvidia-smi
python check_environment.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

You must see your GPU, `"cuda_available": true`, an empty `"warnings"` list, and `True <GPU name>`.

## 5. Tests (run on the CPU on purpose; about 2-10 minutes)

```bash
python -m pytest tests -q
```

Quick version (under a minute):
```bash
python -m pytest tests -q -k "not training and not fine"
```

## 6. Benchmark (speed and memory on your GPU)

```bash
python -m physics_neural_solver.benchmark
```

Result: `outputs/benchmark/<date-time>/summary.md`

## 7. Train (coarse grid, small network)

Training only starts if PyTorch sees a GPU.

```bash
nohup python -m physics_neural_solver.train > train.log 2>&1 &
tail -f train.log
```

`tail -f` only watches the log: Ctrl+C stops the watching, not the training.
The first lines of `train.log` must say `device  cuda  GPU: <your GPU name>`.

While it trains, look at:
```
outputs/runs/coarse_small/losses.png              all losses in one graph
outputs/runs/coarse_small/losses_by_equation.png  one panel per equation
outputs/runs/coarse_small/log.csv                 all the numbers
```

Stop training:
```bash
pkill -f physics_neural_solver.train
```

Continue from the last checkpoint:
```bash
nohup python -m physics_neural_solver.train --resume > train.log 2>&1 &
tail -f train.log
```

## 7b. Or train as a SLURM job

Save this as `train.sbatch` (fill in your partition if the cluster needs one):
```bash
#!/bin/bash
#SBATCH --job-name=neuralP
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
##SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --output=train_%j.log
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi
python -m physics_neural_solver.train
```

```bash
sbatch train.sbatch
squeue -u $USER
tail -f train_<job id>.log
```

## 8. After training

```bash
python -m physics_neural_solver.evaluate --run coarse_small --checkpoint checkpoint_end_of_tilt.pt
python -m physics_neural_solver.evaluate --run coarse_small
python -m physics_neural_solver.export_for_browser --run coarse_small
```

The first line tests the network as it was at the end of the tilt stage (before any stones), the second the final
network. Results:
`outputs/evaluation/coarse_small_end_of_tilt/summary.md`, `outputs/evaluation/coarse_small/summary.md` and
`outputs/export/coarse_small/`

## 9. Send the results back

```bash
git add outputs/benchmark outputs/runs/*/log.csv outputs/runs/*/*.png outputs/evaluation outputs/export/coarse_small
git commit -m "Benchmark and coarse training results"
git push
```

Checkpoints (`*.pt`) are never committed.

## 10. On the Mac: see the trained network in the viewer

```bash
cd "PINNs in graphics/neuralP"
git pull
python3 -m http.server 8001
```

Open in Chrome: http://localhost:8001/classical_solver/?solver=neural&model=coarse_small
(the classical solver: http://localhost:8001/classical_solver/)
