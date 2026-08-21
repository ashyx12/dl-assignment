# Kaggle workflow

Kaggle is the compute environment for this project. GitHub remains the source of truth for code, configs, documentation, and experiment definitions.

## Recommended workflow

1. Create/link a Kaggle Notebook to this GitHub repository.
2. Use the notebook for GPU execution and experiment orchestration.
3. Keep reusable implementation in `src/`; do not build the project as one giant notebook.
4. Install the repository dependencies in the Kaggle session.
5. Save experiment metrics and figures into `results/` when they are small enough for Git.
6. Keep large datasets and model checkpoints out of normal Git history.
7. Save a Kaggle Notebook version after meaningful experiments.
8. Commit the corresponding code/config changes to GitHub as a named project checkpoint.

## Notebook bootstrap

From a Kaggle notebook, the repository can be made available to Python without manually copying source files into the notebook. If the notebook is attached to the repository through Kaggle's GitHub integration, use the repository files directly. Otherwise, the notebook may install the package from GitHub as a fallback.

Example fallback:

```python
!pip install -q "git+https://github.com/ashyx12/dl-assignment.git"
```

This fallback is not intended to be the normal editing workflow. The preferred workflow is to edit reusable code in GitHub and use the Kaggle notebook for execution.

## Checkpoint rule

A Kaggle notebook version is an experiment snapshot. A GitHub checkpoint is the durable project milestone.

Use commit messages such as:

```text
checkpoint: 01 MiniGrid integration
checkpoint: 02 trajectory collection
checkpoint: 03 dataset seed split
```

## Large artifacts

Do not commit `.pt`, `.pth`, `.ckpt`, `.safetensors`, generated raw datasets, or other large artifacts to normal Git history. Use an appropriate artifact store such as Kaggle datasets/models or GitHub Releases/LFS when a checkpoint needs to be preserved.
