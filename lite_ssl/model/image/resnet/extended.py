from lite_ssl.config import logger
from lite_ssl.layers import DINOHead
from lite_ssl.layers.cva_head import IdentityHead
from lite_ssl.model.image.resnet.imagenet import ResNet, BasicBlock, Bottleneck


class ProjResNet(ResNet):
    def __init__(
        self,
        *args,
        cva_head_proj=None,
        project_latent=False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Construct projection head
        logger.info(f"{(cva_head_proj is not None)=}, {project_latent=}")
        if (cva_head_proj is not None) and project_latent:
            self.cva_module_proj = DINOHead(in_dim=self.embed_dim, **cva_head_proj)
        else:
            self.cva_module_proj = IdentityHead()


def _resnet(block, layers, **kwargs):
    return ResNet(block, layers, **kwargs)


def proj_resnet_18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def proj_resnet_34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def proj_resnet_50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
