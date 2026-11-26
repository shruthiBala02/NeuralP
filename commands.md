# Commands: coarse run v2 (copy and paste on the cluster)

Run everything from inside the `NeuralP` folder, on a GPU node.

## 1. Get the latest code

```bash
cd NeuralP
git pull
source .venv/bin/activate
```

## 2. Train (about 25 minutes)

```bash
nohup python -m physics_neural_solver.train --run-name coarse_small_v2 > train_v2.log 2>&1 &
tail -f train_v2.log
```

`tail -f` only watches the log: Ctrl+C stops the watching, not the training.
The first lines must say `device  cuda  GPU: <your GPU name>`.

Stop training:
```bash
pkill -f physics_neural_solver.train
```

Continue from the last checkpoint:
```bash
nohup python -m physics_neural_solver.train --run-name coarse_small_v2 --resume > train_v2.log 2>&1 &
tail -f train_v2.log
```

## 3. Evaluate

```bash
python -m physics_neural_solver.evaluate --run coarse_small_v2 --checkpoint checkpoint_end_of_tilt.pt
python -m physics_neural_solver.evaluate --run coarse_small_v2
```

The first line tests the network at the end of the tilt stage (before any stones), the second the final network.

## 4. Export for the browser

```bash
python -m physics_neural_solver.export_for_browser --run coarse_small_v2
```

## 5. Push the results

```bash
git add outputs/runs/coarse_small_v2/log.csv outputs/runs/coarse_small_v2/*.png outputs/evaluation/coarse_small_v2 outputs/evaluation/coarse_small_v2_end_of_tilt outputs/export/coarse_small_v2
git commit -m "Coarse run v2 results"
git push
```

## 6. On the Mac: open it in the viewer

```bash
cd "PINNs in graphics/neuralP"
git pull
python3 -m http.server 8001
```

Open in Chrome: http://localhost:8001/classical_solver/?solver=neural&model=coarse_small_v2
