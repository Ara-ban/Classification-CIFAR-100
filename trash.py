"""train_metric = utils.Train(modell, loss, optim, device)
        train_metric.update(train_lod)
        train_loss = train_metric.value()

        test_metric = utils.Test(modell, loss, device)
        test_metric.update(valid_lod)
        valid_loss, accuracy = test_metric.value()"""

class BatchMetric:
    def __init__(self, model, loss_function, device):
        self.model = model
        self.loss_function = loss_function
        self.device = device
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.num_samples = 0
        self.correct = 0


class Test(BatchMetric):
    def __init__(self, model, loss_function, device):
        super().__init__(model, loss_function, device)

    def update(self, data_loader):
        self.model.eval()
        self.reset()

        with torch.no_grad():
            for input_tensor, target in data_loader:
                input_tensor, target = input_tensor.to(self.device), target.to(
                    self.device
                )

                # Compute the forward propagation
                y_hat = self.model(input_tensor)
                loss = self.loss_function(y_hat, target)
                predicted_targets = y_hat.argmax(dim=1)

                # Update loss and sample count
                self.total_loss += input_tensor.shape[0] * loss.item()
                self.num_samples += input_tensor.shape[0]

                # Update correct and sample count
                self.correct += (predicted_targets == target).sum().item()
                self.num_samples += input_tensor.shape[0]

    def value(self):
        valid_loss = self.total_loss / self.num_samples
        accuracy = self.correct / self.num_samples
        return valid_loss, accuracy




class Train(BatchMetric):
    def __init__(self, model, loss_function, optimizer, device):
        super().__init__(model, loss_function, device)
        self.optimizer = optimizer

    def update(self, data_loader):
        self.model.train()
        self.reset()

        for input_tensor, target in data_loader:
            input_tensor, target = input_tensor.to(self.device), target.to(self.device)

            # Compute the forward propagation
            y_hat = self.model(input_tensor)
            loss = self.loss_function(y_hat, target)

            # Compute the backward propagation and perform optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update loss and sample count
            self.total_loss += input_tensor.shape[0] * loss.item()
            self.num_samples += input_tensor.shape[0]

    def value(self):
        train_loss = self.total_loss / self.num_samples
        return train_loss



class Accuracy(BatchMetric):
    def __init__(self, model, device):
        super().__init__(model, None, device)  # loss function is not used

    def update(self, data_loader):
        self.model.eval()
        self.reset()

        with torch.no_grad():
            for input_tensor, target in data_loader:
                input_tensor, target = input_tensor.to(self.device), target.to(
                    self.device
                )

                # Compute the forward propagation
                y_hat = self.model(input_tensor)
                predicted_targets = y_hat.argmax(dim=1)

                # Update correct and sample count
                self.correct += (predicted_targets == target).sum().item()
                self.num_samples += input_tensor.shape[0]

    def value(self):
        accuracy = self.correct / self.num_samples
        return accuracy



class Accuracy(BatchMetric):
    def __init__(self, model, device):
        super().__init__(model, None, device)  # loss function is not used

    def update(self, data_loader):
        self.model.eval()
        self.reset()

        with torch.no_grad():
            for input_tensor, target in data_loader:
                input_tensor, target = input_tensor.to(self.device), target.to(
                    self.device
                )

                # Compute the forward propagation
                y_hat = self.model(input_tensor)
                predicted_targets = y_hat.argmax(dim=1)

                # Update correct and sample count
                self.correct += (predicted_targets == target).sum().item()
                self.num_samples += input_tensor.shape[0]

    def value(self):
        accuracy = self.correct / self.num_samples
        return accuracy



