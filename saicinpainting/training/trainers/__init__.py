import logging
import torch
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule


def get_training_model_class(kind):
    if kind == 'default':
        return DefaultInpaintingTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    return cls(config, **kwargs)


def load_checkpoint(train_config, path, map_location='cuda', strict=True):
    model: torch.nn.Module = make_training_model(train_config)
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model

def load_4c_checkpoint_for_5c(train_config, path, map_location='cuda', strict=False):
    new_config = train_config
    new_config.generator.input_nc = 5
    model: torch.nn.Module = make_training_model(new_config)

    state = torch.load(path, map_location=map_location)
    input_layer_name = 'generator.model.1.ffc.convl2l.weight'
    if input_layer_name in state['state_dict']:
        input_layer_weights = state['state_dict'][input_layer_name]
        zeros = torch.zeros_like(input_layer_weights[:, :1, :, :])
        input_layer_weights = torch.cat([input_layer_weights, zeros], dim=1)
        state['state_dict'][input_layer_name] = input_layer_weights

    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model
