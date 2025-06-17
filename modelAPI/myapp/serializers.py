from rest_framework import serializers
from .models import Item

class ImageUploadSerializer(serializers.Serializer):
   image = serializers.ImageField()

