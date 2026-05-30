# NeuralCombs — staged pipeline. Each target is a gate; chain with `&&` to queue them.
# Override: make matrix EPOCHS=400 SEEDS=10
EPOCHS ?= 400
SEEDS  ?= 10
WK0    := experiments/m1_wk0

.PHONY: setup repro smoke pilot matrix analyze all clean

setup:                       ## clone+pin AKOrN, install deps
	bash setup.sh

repro:                       ## native CIFAR-10 classification sanity (the codepath we use)
	cd external/akorn && python train_classification.py wk0_smoke --data cifar10 --epochs 1

smoke:                       ## build the ladder + check deterministic eval (CPU ok)
	cd $(WK0) && python ladder.py && python avalanche_backbone.py

pilot:                       ## GPU-hour budget + R6 fraction-active (k-WTA target)
	cd $(WK0) && python budget.py

matrix:                      ## the chained, resumable ladder runs (Stage 2)
	cd $(WK0) && python run_matrix.py --epochs $(EPOCHS) --seeds $(SEEDS)

analyze:                     ## Stage 3: R6-R5 increment + gate decision
	cd $(WK0) && python analyze.py

# Full dependent chain (each only starts if the previous succeeds):
all: setup repro smoke pilot matrix analyze

clean:
	rm -rf $(WK0)/results $(WK0)/__pycache__
