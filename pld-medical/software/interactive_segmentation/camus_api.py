"""CAMUS-only controller with explicit structure and optional threshold."""
from .models import capabilities
from .service import SegmentationService


class CamusAPI:
    name = 'camus'

    def __init__(self, service=None):
        self.service = service or SegmentationService()

    def capabilities(self):
        return capabilities(self.name)

    def predict(self, image, parameters):
        return self.service.process(self.name,image,parameters)

