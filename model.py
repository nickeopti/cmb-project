import pytorch_lightning as pl
import torch
import torch.nn

import r3
import util


class CMBClassifier(pl.LightningModule):
    def __init__(self, activation_function, learning_rate: float = 0.5):
        super().__init__()

        self.learning_rate = learning_rate

        self.elu = torch.nn.ReLU()

        self.resnet = r3.create_resnet(
            model_depth=8,
            n_input_channels=1,
            num_classes=1,
            activation_function=activation_function
        )
        self.classifier = torch.nn.Linear(64, 1)

        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        z: torch.Tensor = self(x)

        loss = self.loss_function(z.flatten(), y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            for image, predictions, comp, labels, boxes in zip(*batch):
                for i, box in boxes:
                    box = [slice(a.item(), b.item()) for a, b in box]
                    image_box = util.extract_image_box(image, box)
                    if image_box is None:
                        predictions[predictions == i] = 0
                    else:
                        prediction = self(image_box.unsqueeze(0)).item()
                        if prediction < 0:
                            predictions[predictions == i] = 0

                        true_label = torch.any(util.extract_image_box(labels, box) > 0)

                tp, fp, fn, n = util.compute_object_metrics(
                    labels.squeeze().numpy().copy(),
                    predictions.squeeze().numpy().copy(),
                )

                self.log("tp", tp, reduce_fx="sum", on_step=True, on_epoch=False)
                self.log("fp", fp, reduce_fx="sum", on_step=True, on_epoch=False)
                self.log("fn", fn, reduce_fx="sum", on_step=True, on_epoch=False)
                self.log("n", n, reduce_fx="sum", on_step=True, on_epoch=False)

                # comparison to input performance
                tp, fp, fn, n = util.compute_object_metrics(
                    labels.squeeze().numpy().copy(), comp.squeeze().numpy().copy()
                )

                self.log("tp_comp", tp, reduce_fx="sum", on_step=True, on_epoch=False)
                self.log("fp_comp", fp, reduce_fx="sum", on_step=True, on_epoch=False)
                self.log("fn_comp", fn, reduce_fx="sum", on_step=True, on_epoch=False)
                self.log("n_comp", n, reduce_fx="sum", on_step=True, on_epoch=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     momentum=0.9,
        #     weight_decay=2e-05
        # )

        return optimizer
