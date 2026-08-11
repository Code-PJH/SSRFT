# JailBench dataset setup

Due to the JailBench dataset's usage requirements, this repository does not redistribute the JailBench CSV files. To obtain the dataset, please contact the [JailBench research team](https://github.com/STAIR-BUPT/JailBench/tree/main).

## Prepare the source files

Place the two original files in this directory:

```text
test_datasets/JailBench/
├── JailBench-seed.csv
└── JailBench.csv
```

Run the following command from the repository root:

```bash
python jailbench_split.py
```

The original files use these schemas:

- `JailBench-seed.csv`: unnamed ID column, `seed`, `一级领域`, `二级领域`
- `JailBench.csv`: unnamed ID column, `query`, `一级领域`, `二级领域`

The script normalizes both schemas to:

```text
q_id,question,一级领域,content_policy_name
```

With the default arguments, the normalized versions replace the two source files in this directory so that the evaluation scripts can read them directly.

## Reproduced files

For `JailBench-seed.csv`, the script selects 18 `content_policy_name` categories for OOD evaluation with `random_state=42`. It assigns every example from those categories to `ood_test.csv`, then stratifies the remaining examples into equal-sized training and ID-test sets. It writes:

- `train.csv`
- `id_test.csv`
- `ood_test.csv`
- `test.csv`, containing the ID-test and OOD-test rows

For `JailBench.csv`, the script uses stratified sampling with `random_state=42` to create:

- `JailBench_sampled_1080.csv`

All generated JailBench CSV files are ignored by Git and must not be redistributed from this repository.
