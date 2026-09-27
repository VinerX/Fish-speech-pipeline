def set_cuda_device_for_thread(torch_module, device) -> None:
    selected_device = torch_module.device(device)
    if selected_device.type == "cuda" and selected_device.index is not None:
        torch_module.cuda.set_device(selected_device)
