from mmengine.config import read_base

with read_base():
    # from opencompass.configs.datasets.needlebench_v2.needlebench_v2_4k.needlebench_v2_multi_reasoning_4k import needlebench_2needle_en_datasets as needlebench_multi_2needle_en_datasets
    # from opencompass.configs.datasets.needlebench_v2.needlebench_v2_4k.needlebench_v2_multi_reasoning_4k import needlebench_4needle_en_datasets as needlebench_multi_4needle_en_datasets

    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_4k.needlebench_v2_single_4k import needlebench_en_datasets as needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench_v2.needlebench_v2_4k.needlebench_v2_multi_retrieval_4k import needlebench_en_datasets as needlebench_parallel_en_datasets

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
