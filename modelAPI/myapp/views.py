from django.shortcuts import render
from rest_framework import status
from rest_framework.viewsets import ViewSet
from rest_framework.decorators import action
from rest_framework.response import Response
from .serializers import ImageUploadSerializer
from PIL import Image
from pix2tex.cli import LatexOCR
class ImageToTextView(ViewSet):
    @action(detail=False, methods=['post'], url_path='upload')
    def imageConverter(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']

            with open("temp_image.jpg", "wb+") as f:
                for chunk in image.chunks():
                    f.write(chunk)

            try:
                img = Image.open("temp_image.jpg")
                model = LatexOCR()
                extractedLatex = model(img) #run ai model on computer`
            except Exception as e:
                return Response({"error", str(e)}, status=500)

            return Response({"text": extractedLatex}, status=200)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

