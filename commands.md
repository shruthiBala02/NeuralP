# Commands: run v5 - medium network, 1,000,000 steps, coarse AND fine at the same time (copy and paste on the cluster)

Run everything from inside the `NeuralP` folder, on a GPU node.

## 1. Get the latest code

```bash
cd NeuralP
git pull
source .venv/bin/activate
```

## 2. Start both trainings

If the v4 runs are still going on the same GPU, stop them first (they would slow everything down):
```bash
pkill -f "run-name coarse_medium_v4"
pkill -f "run-name fine_medium_v4"
```

Coarse grid (2.5 cm):
```bash
nohup python -m physics_neural_solver.train --grid coarse --run-name coarse_medium_v5 > train_coarse_v5.log 2>&1 &
```

Fine grid (1.25 cm):
```bash
nohup python -m physics_neural_solver.train --grid fine --run-name fine_medium_v5 > train_fine_v5.log 2>&1 &
```

If the node has two GPUs, give each run its own: put `CUDA_VISIBLE_DEVICES=0 ` in front of the first command and
`CUDA_VISIBLE_DEVICES=1 ` in front of the second. On one GPU both share it (memory is enough: ~3 GB + ~10 GB).

Watch them (Ctrl+C only stops the watching, not the training):
```bash
tail -f train_coarse_v5.log
tail -f train_fine_v5.log
```

The first lines must say `device  cuda  GPU: <your GPU name>`, `network  medium` and the right grid.
The table shows the real time per step and the time left. Each v5 training step runs 2 network steps, so it is
about 2x slower than v4: rough guess coarse ~2 days, fine about a week; sharing one GPU makes both slower.
You don't have to wait for the end: a kept copy appears every 200,000 steps.

Every 10,000 steps the log prints a short check of the current network, for example:
`check at step 10,000: still water: fastest 0.004 m/s, volume change 0.00 %  |  slosh: period 1.43 s (theory 1.40 s), swing 3.1 cm`
(also written to `outputs/runs/<run>/checks.csv`). Still water should stay near 0 m/s; the slosh period should approach 1.40 s.

Curriculum: calm water until 15k, tilting until 200k, then stones until 1,000,000.
The learning rate restarts every 200k steps, and a permanent copy of the network is kept at the end of each cycle:
```
outputs/runs/<run>/checkpoint_step_200000.pt     end of tilting (no stones yet)
outputs/runs/<run>/checkpoint_step_400000.pt
outputs/runs/<run>/checkpoint_step_600000.pt
outputs/runs/<run>/checkpoint_step_800000.pt
outputs/runs/<run>/checkpoint_step_1000000.pt    the final network
```
(`<run>` = `coarse_medium_v5` or `fine_medium_v5`)

Stop one run (the other keeps going):
```bash
pkill -f "run-name coarse_medium_v5"
pkill -f "run-name fine_medium_v5"
```

Continue from the last checkpoint (saved every 1,000 steps):
```bash
nohup python -m physics_neural_solver.train --grid coarse --run-name coarse_medium_v5 --resume > train_coarse_v5.log 2>&1 &
nohup python -m physics_neural_solver.train --grid fine --run-name fine_medium_v5 --resume > train_fine_v5.log 2>&1 &
```

## 3. Evaluate a kept network (any time after it appears - training keeps running)

```bash
python -m physics_neural_solver.evaluate --run coarse_medium_v5 --checkpoint checkpoint_step_200000.pt
python -m physics_neural_solver.evaluate --run fine_medium_v5 --checkpoint checkpoint_step_200000.pt
```

Change the number for the others (400000, 600000, 800000, 1000000).
Each writes `outputs/evaluation/<run>_step_<number>/summary.md` and `evaluation.png`.

## 4. Export a network for the browser

```bash
python -m physics_neural_solver.export_for_browser --run coarse_medium_v5 --checkpoint checkpoint_step_200000.pt
python -m physics_neural_solver.export_for_browser --run fine_medium_v5 --checkpoint checkpoint_step_200000.pt
```

They are written to `outputs/export/<run>_step_<number>/`.

## 5. Push the results

```bash
git add outputs/runs/*_medium_v5/log.csv outputs/runs/*_medium_v5/checks.csv outputs/runs/*_medium_v5/*.png outputs/evaluation/*_medium_v5_* outputs/export/*_medium_v5_*
git commit -m "Run v5 results"
git push
```

(The checkpoints are large - ~70 MB each - so they are not pushed. The export is enough for the browser.)

## 6. On the Mac: open it in the viewer

```bash
cd "PINNs in graphics/neuralP"
git pull
python3 -m http.server 8001
```

Open in Chrome (change the run and the number as needed):
http://localhost:8001/classical_solver/?solver=neural&model=coarse_medium_v5_step_200000
