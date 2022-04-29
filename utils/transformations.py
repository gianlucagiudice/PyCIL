from torchvision.transforms import transforms

iLogoDet3K_trsf = {
    'common': [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ],
    'test': [],
    'train': []
}
