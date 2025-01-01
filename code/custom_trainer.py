# Since the compute loss function cannot operate properly, we do not use it

# NEED TO FIND A WAY FOR UNIFIED TRAINING!!!

from trl import SFTTrainer

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, lambda_value=1.0 / 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_value = lambda_value

        