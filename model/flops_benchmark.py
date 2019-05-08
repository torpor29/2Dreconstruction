import torch

def add_flops_counting_methods(net_main_module):
	"""
	Adds flops counting functions to an existing model. After that the flops count should
	be activated and the model should be run on an input image.

	:param net_main_module: torch.nn.Module
		Main module containing network
	:return: torch.nn.Module
		Updated main module with new methods/attributes that are used
		to compute flops.
	"""

	# adding additional methods to the existing module object
	# this is done this way so that function has access to self object

	net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
	net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
	net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
	net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

	net_main_module.reset_flops_count()

	net_main_module.apply(add_flops_mask_variable_or_reset)

	return net_main_module

def compute_average_flops_cost(self):
	batches_count = self.__batch_counter__
	flops_sum = 0
	for module in self.modules():
		if isinstance(module, torch.nn.Conv2d):
			flops_sum += module.__flops__

	return flops_sum / batches_count

def start_flops_count(self):

	add_batch_counter_hook_function(self)
	self.apply(add_flops_counter_hook_function)

def stop_flops_count(self):

	remove_batch_counter_hook_function(self)
	self.apply(remove_flops_counter_hook_function)

def reset_flops_count(self):

	add_batch_counter_variables_or_reset(self)
	self.apply(add_flops_counter_variable_or_reset)

def add_flops_mask(module, mask):
	def add_flops_mask_func(module):
		if isinstance(module, torch.nn.Conv2d):
			module.__mask__ = mask

	module.apply(add_flops_mask_func)

def remove_flops_mask(module):
	module.apply(add_flops_mask_variable_or_reset)

def conv_flops_counter_hook(conv_module, input, output):
	# Can have multiple inputs, getting the first one
	input = input[0]

	batch_size = input.shape[0]
	output_height, output_width = output.shape[2:]

	kernel_height, kernel_width = conv_module.kernel_size
	in_channels = conv_module.in_channels
	out_channels = conv_module.out_channels
	groups = conv_module.groups

	# We count multiply-add as 2 flops
	conv_per_position_flops = 2 * kernel_height * kernel_width * in_channels * out_channels / groups

	active_elements_count = batch_size * output_height * output_width

	if conv_module.__mask__ is not None:
		# (b, 1, h, w)
		flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
		active_elements_count = flops_mask.sum()

	overall_conv_flops = conv_per_position_flops * active_elements_count

	bias_flops = 0

	if conv_module.bias is not None:
		bias_flops = out_channels * active_elements_count

	overall_flops = overall_conv_flops + bias_flops

	conv_module.__flops__ += overall_flops


def batch_counter_hook(module, input, output):
	# Can have multiple inputs, getting the first one
	input = input[0]

	batch_size = input.shape[0]

	module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
	module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
	if hasattr(module, '__batch_counter_handle__'):
		return

	handle = module.register_forward_hook(batch_counter_hook)
	module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
	if hasattr(module, '__batch_counter_handle__'):
		module.__batch_counter_handle__.remove()

		del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
	if isinstance(module, torch.nn.Conv2d):
		module.__flops__ = 0


def add_flops_counter_hook_function(module):
	if isinstance(module, torch.nn.Conv2d):

		if hasattr(module, '__flops_handle__'):
			return

		handle = module.register_forward_hook(conv_flops_counter_hook)
		module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
	if isinstance(module, torch.nn.Conv2d):

		if hasattr(module, '__flops_handle__'):
			module.__flops_handle__.remove()

			del module.__flops_handle__


# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
	if isinstance(module, torch.nn.Conv2d):
		module.__mask__ = None


def count_flops(model, batch_size, device, dtype, input_size, in_channels, *params):
	net = model(*params, input_size=input_size)
	# print(net)
	net = add_flops_counting_methods(net)

	net.to(device=device, dtype=dtype)
	net = net.train()

	batch = torch.randn(batch_size, in_channels, input_size, input_size).to(device=device, dtype=dtype)
	net.start_flops_count()

	_ = net(batch)
	return net.compute_average_flops_cost() / 2  # Result in FLOPs
