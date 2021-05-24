
class CallBack:

    def on_setting_host_to_local(self, host_params, **kwargs):
        return host_params

    def on_client_train_begin(self, data, model, **kwargs):
        return data, model

    def on_client_upload_begin(self, model, **kwargs):
        return model

    def on_host_aggregate_begin(self, client_params, **kwargs):
        return client_params

    def on_client_exit(self):
        pass

    def on_host_exit(self):
        pass
