# RP: Network Representations of Single Cells to Understand Alzheimer's Disease

# Setting Up Mamba on the HPC Cluster

## 1. Log in to the cluster

```bash
ssh <your-student-name>@<cluster-address>
```

## 2. Download and install Miniforge

Navigate to your home directory and download the installer:

```bash
cd /home/nfs/<your-student-name>
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

Run the installer:

```bash
bash Miniforge3-Linux-x86_64.sh
```

When prompted for the install location, accept the default (`/home/nfs/<your-student-name>/miniforge3`) or type it in explicitly.

When asked "Do you wish to update your shell profile to automatically initialize conda?", type **yes**. This adds the necessary lines to your `.bashrc`.

After installation, either log out and back in, or run:

```bash
source ~/.bashrc
```

You can verify the installation with:

```bash
mamba --version
```

## 3. Create your environment

```bash
mamba create -n screni python=3.11
```

Activate it:

```bash
mamba activate screni
```

## 4. Install packages

Install any packages you need, for example:

```bash
mamba install numpy pandas matplotlib
```

## 5. Using your environment in SLURM jobs

Before running your SLURM .sbatch job scripts, make sure you have activated your environment.

```bash
(screni) -bash-4.2$ ./your_code.sbatch
```

DAIC documentation can be found here https://daic.tudelft.nl/
Below is an example script with example settings wherein python is called.

```bash
#! /bin/sh
#SBATCH --partition=general 
#SBATCH --qos=medium
#SBATCH --cpus-per-task=34
#SBATCH --mem=24000
#SBATCH --time=12:57:59
#SBATCH --job-name=name
#SBATCH --mail-user=netid
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err # Set name of error log. %j is the Slurm jobId

ml use /opt/insy/modulefiles;

python main.py

```

## 6. Accessing the shared project folder

The shared project data is located at:

```
/tudelft.net/staff-umbrella/ScReNI
```

You can reference this path in your scripts to read input data or write results.

## Useful commands

| Command | Description |
|---|---|
| `mamba activate screni` | Activate the environment |
| `mamba deactivate` | Deactivate the current environment |
| `mamba list` | List installed packages |
| `mamba install <package>` | Install a package |
| `mamba env list` | List all your environments |

## Troubleshooting

- If `mamba` is not found after installation, run `source ~/.bashrc` or log out and back in.
- If you run out of disk space in your home directory, you can clean the package cache with `mamba clean --all`.
