

## Container environment

The project runs inside an Apptainer/Singularity container (`container.sif`) that packages all Python and R dependencies via pixi. The container is not tracked in git -- copy it to the project root.

### Running commands

```bash
apptainer exec --writable-tmpfs --pwd /opt/app --containall \
  --bind src/:/opt/app/src/ \
  --bind data/:/opt/app/data/ \
  --env PYTHONPATH=/opt/app/src \
  container.sif pixi run --manifest-path /opt/app/pixi.toml python your_script.py
```

For SLURM jobs, see the scripts in `slurm/` for examples.

### Building a new container

Rebuild the container **locally** (requires root/sudo) whenever `pixi.toml`, `pixi.lock`, or `install_deps.R` change:

```bash
sudo apptainer build container.sif container_pixi.def
```

Then copy the resulting `container.sif` to the cluster. The lockfile (`pixi.lock`) pins exact versions, so builds are reproducible.
