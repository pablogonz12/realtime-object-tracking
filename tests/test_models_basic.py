"""
Basic unit tests for models module.
"""

import pytest
import os
from pathlib import Path

# Import directly from the models module what we need to test
from src.models import COCO_CLASSES, YOLO_CLS_INDEX_TO_COCO_ID

@pytest.mark.unit
class TestModels:
    """Basic tests for the models module"""
    
    def test_coco_classes_existence(self):
        """Test that COCO_CLASSES is defined correctly"""
        assert COCO_CLASSES is not None, "COCO_CLASSES should be defined"
        assert isinstance(COCO_CLASSES, list), "COCO_CLASSES should be a list"
        assert len(COCO_CLASSES) > 0, "COCO_CLASSES should not be empty"
        assert "person" in COCO_CLASSES, "COCO_CLASSES should contain person"
    
    def test_yolo_to_coco_mapping(self):
        """Test that YOLO_CLS_INDEX_TO_COCO_ID is defined correctly"""
        assert YOLO_CLS_INDEX_TO_COCO_ID is not None, "YOLO_CLS_INDEX_TO_COCO_ID should be defined"
        assert isinstance(YOLO_CLS_INDEX_TO_COCO_ID, dict), "YOLO_CLS_INDEX_TO_COCO_ID should be a dictionary"
        assert len(YOLO_CLS_INDEX_TO_COCO_ID) > 0, "YOLO_CLS_INDEX_TO_COCO_ID should not be empty"
        # Check if some common class indices are mapped correctly
        assert 0 in YOLO_CLS_INDEX_TO_COCO_ID, "YOLO_CLS_INDEX_TO_COCO_ID should contain key 0"
        assert isinstance(YOLO_CLS_INDEX_TO_COCO_ID[0], int), "YOLO_CLS_INDEX_TO_COCO_ID values should be integers"
