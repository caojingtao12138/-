from functools import partial
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule
from braincog.datasets.datasets import get_cifar10_data

### Create Model

@register_model
class SNN4(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encoder_type='direct',
                 *args,
                 **kwargs):
            super().__init__(step, encoder_type, *args, **kwargs)
            self.num_classes = num_classes

            self.node = node_type

            self.dataset = kwargs['dataset'] if 'dataset' in kwargs else 'dvsc10'
            # is_dvs_data
            init_channel = 3

            self.feature = nn.Sequential(
                BaseConvModule(init_channel, 16, kernel_size=(3,3), padding=(1,1), node=self.node),
                BaseConvModule(16, 64, kernel_size=(5,5), padding=(2,2), node=self.node),
                nn.AvgPool2d(2),
                BaseConvModule(64, 128, kernel_size=(5,5), padding=(2,2), node=self.node),
                nn.AvgPool2d(2),
                BaseConvModule(128, 256, kernel_size=(5,5), padding=(2,2), node=self.node),
                nn.AvgPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, self.num_classes),
            )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        # Layer by layer
        x = self.feature(inputs)
        print(x.shape)
        x = self.fc(x)
        x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
        return x

train_loader, _, _, _ = get_cifar10_data(batch_size=64, num_workers=0, step=8)
it = iter(train_loader)
inputs, labels = it.__next__()
print(inputs.shape, labels.shape)

model = SNN4(layer_by_layer=True).cuda()
print(model)
print(type(train_loader))
outputs = model(inputs.cuda())
print(outputs)