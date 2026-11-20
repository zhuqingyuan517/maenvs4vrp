import torch


class NonFiniteTrainingError(FloatingPointError):
    """Raised before a non-finite quantity reaches a sampling or update op."""


def _describe_invalid(name, tensor, invalid, context=None):
    detached = tensor.detach()
    invalid_count = int(invalid.sum().item())
    if invalid_count == 0:
        return f"{name} is valid"

    valid_values = detached[~invalid]
    if valid_values.numel():
        valid_min = float(valid_values.min().item())
        valid_max = float(valid_values.max().item())
        value_range = f"valid_min={valid_min:.6g}, valid_max={valid_max:.6g}"
    else:
        value_range = "no valid values"

    locations = torch.nonzero(invalid, as_tuple=False)[:8].cpu().tolist()
    message = (
        f"{name} contains {invalid_count}/{detached.numel()} invalid values; "
        f"shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"{value_range}, first_locations={locations}"
    )
    if context:
        message = f"{message}; {context}"
    return message


def describe_nonfinite(name, tensor, context=None):
    detached = tensor.detach()
    return _describe_invalid(
        name,
        detached,
        ~torch.isfinite(detached),
        context=context,
    )


def assert_finite(name, tensor, context=None):
    if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return
    if torch.isfinite(tensor).all():
        return
    raise NonFiniteTrainingError(describe_nonfinite(name, tensor, context))


def assert_not_nan_or_posinf(name, tensor, context=None):
    if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return
    invalid = torch.isnan(tensor) | torch.isposinf(tensor)
    if not invalid.any():
        return
    raise NonFiniteTrainingError(_describe_invalid(name, tensor, invalid, context))


def tensor_or_module_is_finite(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or torch.isfinite(value).all()
    return True
