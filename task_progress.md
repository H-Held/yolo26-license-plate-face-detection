 YOLO26m@640 Training Run - Task Progress

- [ ] Fix: discovered nvidia-smi shows ~97% VRAM (not 85%) — batch finder measured PyTorch reserved instead of true board usage
- [ ] Stop current training + watchdog (prevent auto-retry)
- [ ] Patch batch_finder.py to measure true nvidia-smi peak VRAM per GPU during DDP probes
- [ ] Re-probe batch size under real DDP with nvidia-smi measurement; target ≤85% on all 3 GPUs
- [ ] Relaunch training with corrected batch size
- [ ] Verify epoch 1 runs with GPU_mem ≤85% on all cards
- [ ] Start Windows Program 2 resume watchdog for cross-session protection
- [ ] Update memory with final run configuration