from .call_back_base import CallBack


class DLGAttack(CallBack):

    def __init__(self):
        self.client_params_list = []

    def on_host_aggregate_begin(self, client_params, **kwargs):
        self.client_params_list.append(client_params)
        return client_params

    def on_host_exit(self):
        pass
