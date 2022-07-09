from models.discriminator import conv

discriminator_dict = {
    'dc': conv.DCDiscriminator,
    'resnet': conv.DiscriminatorResnet,
}
