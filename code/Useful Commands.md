# Useful Commands
 
```table-of-contents
title: ## Table of Contents
minLevel:2
```
## Setup jobs

### Transfer Datasets
```
rsync -avz --progress path/to/dataset alan:~/path/to/directory/
ex:
rsync -avz --progress code/datasets/openai_eval_data alan:~/datasets/openai/
```
### Transfer files to alan cluster
```
scp path/to/file alan:~/path/to/dirctory/
ex:
scp code/slurm_jobs/run_test.sbatch alan:~/code/slurm_jobs/
```

### Connect to Alan cluster
```
ssh alan
```

### Schedule Slurm job
```
sbatch path/to/file
ex:
sbatch ~/code/slurm_jobs/run_test.sbatch
```
This will print the job ID.
## Manage jobs

### List all running jobs
```
squeue -u $USER
```
### Monitor live 
```
tail -f path/to/output-log-file.out
ex:
tail -f code/logs/test_[JOBID].out
```
### After job completion

#### For output
```
more path/to/output-log-file.out
```
#### For errors
```
more path/to/output-log-file.err
```
