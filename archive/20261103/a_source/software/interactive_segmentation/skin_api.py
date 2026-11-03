"""Skin-only controller. No CAMUS or classification state is accepted here."""
from .models import capabilities
from .service import SegmentationService


class SkinAPI:
    name = 'skin'

    def __init__(self, service=None):
        self.service = service or SegmentationService()

    def capabilities(self):
        return capabilities(self.name)

    def predict(self, image, parameters):
        return self.service.process(self.name,image,parameters)

