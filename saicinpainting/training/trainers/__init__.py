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

def make_4output_from_3output(config, path, map_location='cuda'):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make 4 output training model {kind} from 3 output model in {path}')

    cls = get_training_model_class(kind)
    model: torch.nn.Module = cls(config, **kwargs)

    state = torch.load(path, map_location=map_location)

    keys = 'discriminator.model0.0.weight \
    discriminator.model0.0.bias \
    discriminator.model1.0.weight \
    discriminator.model1.0.bias \
    discriminator.model1.1.weight \
    discriminator.model1.1.bias \
    discriminator.model1.1.running_mean \
    discriminator.model1.1.running_var \
    discriminator.model1.1.num_batches_tracked \
    generator.model.15.weight \
    generator.model.15.bias \
    generator.model.16.weight \
    generator.model.16.bias \
    generator.model.16.running_mean \
    generator.model.16.running_var \
    generator.model.16.num_batches_tracked \
    generator.model.18.weight \
    generator.model.18.bias \
    generator.model.19.weight \
    generator.model.19.bias \
    generator.model.19.running_mean \
    generator.model.19.running_var \
    generator.model.19.num_batches_tracked \
    generator.model.21.weight \
    generator.model.21.bias \
    generator.model.22.weight \
    generator.model.22.bias \
    generator.model.22.running_mean \
    generator.model.22.running_var \
    generator.model.22.num_batches_tracked \
    generator.model.25.weight \
    generator.model.25.bias'.split(' ')

    for key in keys:
        state['state_dict'].pop(key, None)

    model.load_state_dict(state['state_dict'], strict=False)

    return model
    


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
