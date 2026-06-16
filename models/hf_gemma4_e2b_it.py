from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma-4-e2b-it',
        path='google/gemma-4-e2b-it',
        max_seq_len=131072,
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<turn|>'],
        model_kwargs=dict(
            dtype='bfloat16',
            device_map='cuda',
        ),
    )
]
