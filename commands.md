# Commands: coarse run v3 (copy and paste on the cluster)

Run everything from inside the `NeuralP` folder, on a GPU node.

## 1. Get the latest code

```bash
cd NeuralP
git pull
source .venv/bin/activate
```

## 2. Train (about 25 minutes)

```bash
nohup python -m physics_neural_solver.train --run-name coarse_small_v3 > train_v3.log 2>&1 &
tail -f train_v3.log
```

`tail -f` only watches the log: Ctrl+C stops the watching, not the training.
The first lines must say `device  cuda  GPU: <your GPU name>`.

Stop training:
```bash
pkill -f physics_neural_solver.train
```

Continue from the last checkpoint:
```bash
nohup python -m physics_neural_solver.train --run-name coarse_small_v3 --resume > train_v3.log 2>&1 &
tail -f train_v3.log
```

## 3. Evaluate

```bash
python -m physics_neural_solver.evaluate --run coarse_small_v3 --checkpoint checkpoint_end_of_tilt.pt
python -m physics_neural_solver.evaluate --run coarse_small_v3
```

The first line tests the network at the end of the tilt stage (before any stones), the second the final network.
Each writes `summary.md` and `evaluation.png` (still water, sloshing and splash over time).

## 4. Export for the browser

```bash
python -m physics_neural_solver.export_for_browser --run coarse_small_v3
```

## 5. Push the results

```bash
git add outputs/runs/coarse_small_v3/log.csv outputs/runs/coarse_small_v3/*.png outputs/evaluation/coarse_small_v3 outputs/evaluation/coarse_small_v3_end_of_tilt outputs/export/coarse_small_v3
git commit -m "Coarse run v3 results"
git push
```

## 6. On the Mac: open it in the viewer

```bash
cd "PINNs in graphics/neuralP"
git pull
python3 -m http.server 8001
```

Open in Chrome: http://localhost:8001/classical_solver/?solver=neural&model=coarse_small_v3
