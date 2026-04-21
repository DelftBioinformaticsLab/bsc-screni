## Container environment

The project runs inside an Apptainer/Singularity container that packages all Python dependencies via pixi. The container is not tracked in git — copy it to the project root on the cluster.

### Running commands

```bash
apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --bind output/:/opt/app/output/ \
  --env PYTHONPATH=/opt/app/src \
  container_0-1-2.sif pixi run --manifest-path /opt/app/pixi.toml python your_script.py
```

For SLURM jobs, see the scripts in `slurm/` for examples.

### Building a new container

To build a container, you need to install apptainer on your local machine.
Either natively in Linux, or in WSL on Windows (https://apptainer.org/docs/admin/main/installation.html#installation-on-windows-or-mac)

Rebuild the container **locally** whenever `pixi.toml` or `pixi.lock` change:

```bash
pixi lock
sudo apptainer build container_X-Y-Z.sif container_pixi.def
```

Then copy to the cluster:

```bash
wsl bash -c "scp /mnt/c/.../container_X-Y-Z.sif daic:/tudelft.net/staff-umbrella/ScReNI/bsc-screni/"
```

Container versions:
- `container.sif` — original (v0.1.0), base scanpy/seurat deps only
- `container_0-1-1.sif` — added harmonypy, muon, celltypist
- `container_0-1-2.sif` — removed R dependencies, Python-only

The lockfile (`pixi.lock`) pins exact versions, so builds are reproducible.

### Local development

Some processing steps can run locally via pixi tasks:

```bash
pixi run load-paper        # Phase 0: load paper's retinal data
pixi run process-pbmc      # Phase 0: PBMC loading + annotation
pixi run integrate-pbmc    # Phase 1: PBMC WNN integration
pixi run feature-select    # Phase 2: feature selection
pixi run gene-peak         # Phase 3: gene-peak-TF relations
```

Note: CellTypist annotation crashes on Windows due to a numpy int32 overflow
with large sparse matrices. Use the cluster for PBMC Phase 0.
