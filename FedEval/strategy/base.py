
class FedModel:

    # Host functions
    def update_host_params(self):
        # Do : Aggregate client params
        pass

    # Client functions
    def set_host_params_to_local(self):
        # Do : Set the host params to client local
        pass

    # Client functions
    def fit_on_local_data(self):
        # Do : Clients' local training
        pass
    
    # Client functions
    def retrieve_local_upload_info(self):
        # Do : CLients retrive the uploading params
        pass

    # Client functions
    def local_evaluate(self):
        # Do : Clients' local evaluate
        pass


class CallBack:

    def on_setting_host_to_local(self, host_params, **kwargs):
        return host_params

    def on_client_train_begin(self, data, model, **kwargs):
        # Do : Data Poisoning Attacks
        return data, model

    def on_client_upload_begin(self, model, **kwargs):
        # Do : Model Poisoning Attacks
        return model

    def on_host_aggregate_begin(self, client_params, **kwargs):
        # Do : Verification to defence poisoning attacks
        # Do : Gradients attacks
        return client_params

    def on_client_exit(self):
        # Do : E.g., clients local finetunes after the training
        pass

    def on_host_exit(self):
        # Do : Model Inversion Attacks
        pass


